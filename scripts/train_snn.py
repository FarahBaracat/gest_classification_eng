from srcs.network import Net, EMGCustomDataset
from srcs.engdataset import ENGDataset, Nerve

import utils.evaluate_model as meval #forward_test_data, log_mean_conf_matrix, compute_metric
from utils.preprocess_spikes import dense_to_sparse, split_rep_tensor_into_bins
from utils.load_files import load_encoded_data
from utils.plotting import plot_vmem
import utils.plot as uplot 

from constants import COLOR_DICT, MAX_REP_DUR, CLASS_TO_GEST
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from snntorch import functional as SF
import torch
import wandb
from collections import namedtuple, Counter
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from brian2 import *
import random
import os
from constants import *

plt.rcParams.update({"figure.dpi": 150})
plt.rcParams['axes.axisbelow'] = True
matplotlib.rcParams.update({'font.size': 6})
logging.getLogger().setLevel(logging.INFO)
matplotlib.rc('axes', edgecolor=COLOR_DICT['clouds'], linewidth=0.4)


# Some constants: TODO: move to a data config (data class)
LABEL_COL = 'label'

LOG_EVERY_N_EP = 10  # plot c1onfusion matrix every n epochs


def train():
    rec_day = 16
    session = '01'
    save_snn_figs = True

    if not os.path.exists(SNN_FIG):
        os.makedirs(SNN_FIG)

    eng_dataset = ENGDataset(day= rec_day, session=session, load_raw_data = False, save_figs=False)
    bad_channels = []  #currently identified based on mean and std: [1,2,5,7,43]   in previous implmentation: [1,2,43]

    # torch parameters
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
    params_defaults = dict(n_epochs=1, # default 150
                       batch_size=16,  # default 16
                       tau_mem=54e-3,  # 50e-3,
                       tau_syn=10e-3,  # 30e-3,
                       learn_tausyn  = False,
                        learn_taumem = False,
                       lif_type='synaptic',
                       dt=0.01,
                       k_cv=5, 
                       seed=10,
                       lr_rate=0.01,
                       correct_rate=0.95,  # 150 Hz for 200 ms/ 98 Hz for 150 ms/ 97 Hz for 100 ms
                       incorrect_rate=0,
                       bin_width=0.2,

                       monitor_indices=[5, 10, 20, 35],
                       enc_vth=0.02, # default 0.05
                       enc_tau_mem=0.03, #default 0.020,  
                       w_init_mean=0.3, # default 0.6,
                       w_init_std=0.1,   
                       w_init_a=-0.1,
                       w_init_b=0.1,
                       w_init_dist='normal',#'normal',
                       split_rep=True,
                       split_type='with_overlap',
                       overlap_perc=0.5,
                       )

    wandb.init(project="revisit_eng_2023", entity="farahbaracat", config=params_defaults)

    # fix seeds
    torch.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    random.seed(wandb.config.seed)

    
    # load encoded data defined by neuron parameters
    Gest = namedtuple('gesture', ['id', 'phase'])
    sel_gest = [Gest(0, 'ext'), Gest(1, 'flx'), Gest(3, 'flx'), Gest(4, 'flx')]
    spikes = load_encoded_data(wandb.config['enc_tau_mem'], wandb.config['enc_vth'], sel_gest, 
                               file_prefix='engfullwave_to_spikes',
                               file_suffix=f'nofeature_remove_{bad_channels}')

    # Network Architecture
    num_outputs = len(sel_gest)
    num_inputs = eng_dataset.n_channels
    rep_num_steps = int(MAX_REP_DUR/wandb.config['dt'])
    # wind_num_steps = int(wandb.config['']/wandb.config['dt'])
    print(f"Input size: {num_inputs}, Output size: {num_outputs}")

    # Prepare the dataset: sparse array at each dt:each sample should be (timestep, neuron_id)?
    # Temporal Dynamics: discretization of time
    input, labels_before, rep_num_steps = dense_to_sparse(spikes, sel_gest, rep_num_steps,
                                                      num_inputs, MAX_REP_DUR, wandb.config['dt'], device, dtype)


    logging.info(f"Discritized time: {wandb.config['dt']}, Number of steps: {rep_num_steps}")
    print(f"\n Scaled correct rate:{wandb.config['correct_rate']/rep_num_steps}")
    loss_fn = SF.mse_count_loss(correct_rate=wandb.config['correct_rate'],
                                incorrect_rate=wandb.config['incorrect_rate'])

    kf = StratifiedKFold(n_splits=wandb.config['k_cv'], random_state=wandb.config.seed, shuffle=True)

    # split the repetitions first before splitting further into time bins
    kf_split = kf.split(np.arange(input.size(0)), labels_before)
    if wandb.config['split_rep']:
        # num_steps can change if split_rep is True
        input, labels, rep_num_steps, n_split_bins = split_rep_tensor_into_bins(
            input, labels_before, rep_num_steps, bin_width=wandb.config['bin_width'])
    print(f"\n\nInput size after splitting into bins: {input.size()}\n")
    
    # plotting the spikes per label
    # plot_mean_firing_rate_temporal(input, labels)
    dataset = EMGCustomDataset(input, labels)  # full dataset with the time bins

    epochs_loss_train = torch.empty(wandb.config['n_epochs'], wandb.config['k_cv'])
    epochs_loss_test = torch.empty_like(epochs_loss_train)
    epochs_acc_test = torch.empty_like(epochs_loss_train)
    epochs_acc_train = torch.empty_like(epochs_loss_train)

    conf_matrix_fold = np.empty((wandb.config['n_epochs'],  wandb.config['k_cv'], num_outputs, num_outputs))

    report_df = pd.DataFrame()  # iteratively add to the classification report dataframe
    prec_matrix_fold = np.empty((wandb.config['n_epochs'],  wandb.config['k_cv'], num_outputs))
    rec_matrix_fold = np.empty_like(prec_matrix_fold)

    metrics_w_dict = {'acc_val': [],  'f1_val': [],
                      'prec_val': [], 'recall_val': []}

    for fold_i, (train_ind, test_ind) in enumerate(kf_split):
        print(f"fold#:{fold_i} train id:{train_ind}   test_ind:{test_ind}")

        # map train_ind and test_ind to new ids after splitting into bins
        if wandb.config['split_rep']:
            old_train_ind, old_test_ind = train_ind, test_ind
            train_ind, test_ind = map_new_ids(train_ind, test_ind, n_split_bins)
            print(
                f"New ids after expanding into bins\n-------\nfold#:{fold_i} train id:{train_ind}   test_ind:{test_ind}\n")
            print(
                f"Train: {Counter(labels[train_ind].detach().numpy())}  Test:{Counter(labels[test_ind].detach().numpy())}")
        # create dataloader
        train_sub = Subset(dataset, train_ind)
        test_sub = Subset(dataset, test_ind)
        monitor_sub = Subset(dataset, wandb.config['monitor_indices'])

        train_loader = DataLoader(train_sub, batch_size=wandb.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_sub, batch_size=wandb.config['batch_size'], shuffle=True)
        monitor_loader = DataLoader(monitor_sub, batch_size=wandb.config['batch_size'], shuffle=False)

        # initialize the network
        net = Net(rep_num_steps, num_inputs, num_outputs).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config['lr_rate'], betas=(0.9, 0.999))

        # read the weights before training
        w_init = net.fc1.weight.detach().clone().numpy()
        vth_init = net.lif1.threshold.detach().clone().numpy()

        for epoch in range(wandb.config['n_epochs']):
            e_train_loss = 0
            tr_acc = 0
            total_samp = 0
            # Training loop
            for data, targets in iter(train_loader):
                spk_in = data.to(device)
                targets = targets.to(device)
                logging.debug(f"data:{data.size()} targets:{torch.unique(targets)}")
                logging.debug(f"check on input spikes:{spk_in.size()}")

                # forward pass
                net.train()
                rec = net.forward(spk_in)
                logging.debug(f"check on target:{targets.size()}   spk_rec:{rec['spk1'].size()}")
                loss_train = loss_fn(rec['spk1'], targets)

                # Update weights 
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # Accumulate the loss per item
                e_train_loss += loss_train.item()

                # Log train acc
                tr_acc += SF.accuracy_rate(rec['spk1'], targets) * wandb.config['batch_size']
                total_samp += wandb.config['batch_size']

            if epoch % LOG_EVERY_N_EP == 0:
                _, _, mon_rec, _, _ = meval.forward_test_data(monitor_loader, net, device, loss_fn, return_rec=True)
                fig = plot_vmem(mon_rec, labels[wandb.config['monitor_indices']], epoch, fold_i)
                wandb.log({f"monitor_vmem": wandb.Image(fig.get_figure())})

            epochs_loss_train[epoch, fold_i] = e_train_loss/len(train_ind)
            epochs_acc_train[epoch, fold_i] = tr_acc/total_samp

            # Testing loop
            test_acc, test_loss, y_pred, y_true = meval.forward_test_data(test_loader, net, device, loss_fn)
            epochs_acc_test[epoch, fold_i] = test_acc.item()
            epochs_loss_test[epoch, fold_i] = test_loss.item()

            # Get confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")

            print(f"{conf_matrix.shape}")

            conf_matrix_fold[epoch, fold_i, :, :] = conf_matrix

            # compute precision and recall per class
            prec, rec, _, _ = meval.compute_metric(
                y_true, y_pred, 'precision_recall_fscore_support', CLASS_TO_GEST.values(), avg=None)
            prec_matrix_fold[epoch, fold_i, :] = prec
            rec_matrix_fold[epoch, fold_i, :] = rec

            if epoch == wandb.config['n_epochs']-1:
                metrics_w_dict["acc_val"].append(meval.compute_metric(y_true, y_pred, 'acc', CLASS_TO_GEST.values()))
                metrics_w_dict["f1_val"].append(meval.compute_metric(y_true, y_pred, 'f1', CLASS_TO_GEST.values()))

                prec, recall, _, _ = meval.compute_metric(
                    y_true, y_pred, 'precision_recall_fscore_support', CLASS_TO_GEST.values())
                metrics_w_dict["prec_val"].append(prec)
                metrics_w_dict["recall_val"].append(recall)

            if epoch % LOG_EVERY_N_EP == 0:
                conf_fig = plot_confusion_matrix(y_true, y_pred, CLASS_TO_GEST, title=f'fold {fold_i}, epoch {epoch}', return_fig=True)
                wandb.log({f"confusion_matrix": wandb.Image(conf_fig.get_figure())})

                # compute classification report
                report = meval.compute_metric(y_true, y_pred, 'clf_report', CLASS_TO_GEST.values())
                tmp_df = pd.DataFrame(report)
                tmp_df['epoch'] = epoch
                tmp_df['fold'] = fold_i

                # add to the dataframe
                report_df = pd.concat([report_df, tmp_df], axis=0)

            print(
                f"Epoch [{epoch}]  - Train Loss:{epochs_loss_train[epoch, fold_i]}    Test Loss:{epochs_loss_test[epoch, fold_i]}")
            print(
                f"Train Acc: {epochs_acc_train[epoch, fold_i] * 100:.2f}%    Test Acc: {epochs_acc_test[epoch, fold_i] * 100:.2f}%\n")
            wandb.log({'test_acc': test_acc, 'test_loss': test_loss,
                      'train_acc': epochs_acc_train[epoch, fold_i], 'train_loss': e_train_loss/len(train_ind), 'fold': fold_i})
            wandb.log({'Neuron threshold': net.lif1.threshold.detach().clone()})
    wandb.log({'avg_ep_ts_acc': epochs_acc_test[-1, :].mean().item(),
              'avg_ep_tr_acc': epochs_acc_train[-1, :].mean().item()})

    # average the confusion matrix over all folds: returns array (n_epochs, n_classes, n_classes)
    mean_conf_matrix, std_conf_matrix = compute_mean_std_for_array(conf_matrix_fold, axis=1)
    print(f"mean pred per class {mean_conf_matrix[-1,:,:]}")
    print(f"std pred per class {std_conf_matrix[-1,:,:]}")
    print("\n ----------------")
    wandb.log({'mean_conf_diagonal': np.mean(np.diag(mean_conf_matrix[-1,:,:]))})

    # take the mean over k-folds
    mean_prec, std_prec = compute_mean_std_for_array(prec_matrix_fold, axis=1)
    mean_recall, std_recall = compute_mean_std_for_array(rec_matrix_fold, axis=1)
    results_df = pd.DataFrame(metrics_w_dict)

    # read the weights and vth after training
    w_trained = net.fc1.weight.detach().clone().numpy()
    vth_trained = net.lif1.threshold.detach().clone().numpy()

    fig = uplot.plot_acc_weighted_prec_recall(results_df, plot_train=False)
    wandb.log({f"weighted precision and recall": wandb.Image(fig)})

    # Plot metrics 
    # Mean confusion matrix across folds
    uplot.log_mean_conf_matrix(eng_dataset.day, mean_conf_matrix, save_fig=save_snn_figs,  cmap='PuBuGn')

    # barplot correct predictions per class + sd for the last epoch
    uplot.plot_mean_acc_per_class(num_outputs, mean_conf_matrix[-1, :, :], std_conf_matrix[-1, :, :])

    # barplot precision and recall per class + sd for the last epoch
    uplot.plot_mean_prec_rec_per_class(num_outputs, mean_prec[-1], std_prec[-1], mean_recall[-1], std_recall[-1])

    # plot the learned weights and thresholds
    uplot.plot_learnt_wdist(eng_dataset.day, w_trained, w_init, save_fig=save_snn_figs)
    uplot.plot_learnt_wheat(eng_dataset.day, w_trained, save_fig=save_snn_figs, cmap='PuBuGn')
    uplot.plot_learnt_threshold(eng_dataset.day, vth_trained, vth_init, save_fig=save_snn_figs)

    wandb.log({f"Classification Report": wandb.Table(dataframe=report_df)})
    print(f"results_df:{results_df}\n\n")




def compute_mean_std_for_array(metric_matrix, axis):

    mean_metric = metric_matrix.mean(axis=axis)
    std_metric = metric_matrix.std(axis=axis)
    return mean_metric, std_metric


def map_new_ids(train_ind, test_ind, n_split_bins):
    """
    map the indices of the train and test sets to new indices after splitting the dataset into bins
    ARE you considering that some labels have less bins??
    """
    train_ind_new = np.concatenate([np.arange(i*n_split_bins, (i+1)*n_split_bins) for i in train_ind])
    test_ind_new = np.concatenate([np.arange(i*n_split_bins, (i+1)*n_split_bins) for i in test_ind])

    return train_ind_new, test_ind_new


if __name__ == '__main__':
    train()
