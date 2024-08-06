from constants import CLASS_TO_GEST, COLOR_DICT

import plotly.io as plt_io
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import wandb
import snntorch.spikeplot as splt
import torch
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import pandas as pd
from constants import *
import utils.preprocess_analog as pre
import utils.plot as uplot
# from tueplots import figsizes

plt.rcParams.update({"figure.dpi": 150})
# plt.rcParams.update(figsizes.neurips2021(nrows=15, ncols=20, height_to_width_ratio=1))
plt.rcParams['axes.axisbelow'] = True
matplotlib.rc('axes', edgecolor=COLOR_DICT['clouds'], linewidth=0.4)

matplotlib.rcParams.update({'font.size': 6})


def plot_3d(component1, component2, component3, l):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10,
            color=l,                # set color to an array/list of desired values
            colorscale='Rainbow',   # choose a colorscale
            opacity=1,
            line_width=1
        )
    )])
    # tight layout
    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50),
                      width=1300, height=1000)
    fig.layout.template = 'plotly_dark'

    fig.show()


def plot_distribution(tensor, title, xlabel='Values', ylabel='Frequency', bins=100):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    ax.hist(tensor.flatten(), bins=bins, color=COLOR_DICT['wisteria'])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    return fig

def plot_heatmap(tensor, title, ax, cbar_label=None, vmin=None, vmax=None, cbar=False, ylabel= 'Channels', xlabel='Reps'):
    hmap = sns.heatmap(tensor,ax=ax, cbar_kws={'label': cbar_label}, cmap='rocket_r', vmin=vmin, vmax=vmax, cbar=cbar)
    ax.set_title(title, fontsize=10)

    ax.set_yticks(np.arange(0,tensor.size(0) ,5), minor=False)

    # hmap.set_yticklabels(list(tensor.index)[::5], rotation=0, fontsize=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    return hmap



def log_weights(weights, is_trained=False):
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)

    if is_trained:
        title = 'After training'
        prefix = 'trained'
    else:
        title = 'Before training'
        prefix = 'untrained'  

    weights_df = pd.DataFrame(weights.T)
    uplot.plot_heatmap(weights_df, ax=ax, cbar=False, cbar_label='Weight (a.u.)',
                   yticks_label=weights_df.index,
                   xticks_label=weights_df.columns, yticks_step=5, xticks_step=1,
                  title="",ylabel='Input channels', 
                  xlabel='Output neurons', cmap='YlGnBu')  
    # hmap = plot_heatmap(w_tensor.T, title, ax, cbar_label='weights', vmin=None, vmax=None, cbar=True, 
    #                     ylabel='Input Neurons', 
    #                     xlabel='Output Neurons')

    hist = plot_distribution(weights, title, xlabel='Weights', ylabel='Frequency', bins=100)
    
    wandb.log({f"{prefix}_w_hmap": wandb.Image(hmap.get_figure())})
    wandb.log({f"{prefix}_w_hist": wandb.Image(hist)})



def plot_vmem(mon_rec, targets, epoch, fold_i):
    fig = plt.figure(figsize=(15,7))
    
    n_cols = len(wandb.config['monitor_indices'])

    gs = gridspec.GridSpec(ncols=n_cols, nrows=5)
    for sample in range(n_cols):
        if wandb.config['batch_size'] == 1:
            rec = mon_rec[sample]
            index = 0
        else:
            rec = mon_rec[0]
            index = sample
        spk_rec , mem_rec, cur_rec, spk_in_rec = rec['spk1'], rec['mem1'], rec['cur1'], rec['spk_in']

        
        ax = fig.add_subplot(gs[0,sample])
        splt.raster(spk_in_rec[:,index,:], ax, s=2)
        plt.title(f'Input Spikes label {targets[sample]}')
        plt.box(False)

        ax = fig.add_subplot(gs[1,sample])
        splt.raster(spk_rec[:,index,:], ax, s=2)
        plt.ylabel(f'Output Spikes')
        plt.box(False)

        ax = fig.add_subplot(gs[2,sample])
        plt.plot(mem_rec[:,index,:].cpu().detach().numpy(), label='membrane')   
        plt.ylabel('v_mem')
        plt.box(False)

        ax = fig.add_subplot(gs[3,sample])
        plt.plot(cur_rec[:,index,:].cpu().detach().numpy(), label='syn')
        plt.ylabel('Input. Curr')
        plt.box(False)

        if 'syn1' in rec.keys():
            syn_rec = rec['syn1']
            ax = fig.add_subplot(gs[4,sample])
            plt.plot(syn_rec[:,index,:].cpu().detach().numpy(), label='syn')
            plt.ylabel('Syn. Curr')
            plt.box(False)


        plt.tight_layout()
    fig.text(0.5, 0.99, f'Epoch {epoch}  fold:{fold_i}', ha='center', va='center', fontsize=10)

    return fig

   

# def plot_mean_acc_per_class(num_outputs, mean_conf_matrix_val, std_conf_matrix_val, mean_conf_matrix_train=None, std_conf_matrix_train=None, class_to_gest=CLASS_TO_GEST, wandb_log=True):
    
#     bwidth= 0.2
#     capsize= 2
#     fig = plt.figure(figsize=(4,2))
    
    
#     mean_cor_pred_val = np.diag(mean_conf_matrix_val) # average is across the folds
#     std_cor_pred_val = np.diag(std_conf_matrix_val)  # sd is across the folds

#     print(f"Mean corr pred val: {mean_cor_pred_val}\n std corr pred val: {std_cor_pred_val}")

#     plt.bar(np.arange(num_outputs) + bwidth, mean_cor_pred_val, yerr=std_cor_pred_val,
#                 width=bwidth, color=COLOR_DICT['viz_orange'],
#                 capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='val')
    
#     if mean_conf_matrix_train is not None: # used for plotting the train accuracy if provided, this is the case with the SVM analysis
#         mean_cor_pred_train = np.diag(mean_conf_matrix_train) 
#         std_cor_pred_train = np.diag(std_conf_matrix_train)  
#         print(f"Mean corr pred train: {mean_cor_pred_train}\n std corr pred train: {std_cor_pred_train}")

#         plt.bar(np.arange(num_outputs), mean_cor_pred_train, yerr=std_cor_pred_train,
#             width=bwidth, color=COLOR_DICT['wisteria'],
#             capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='train')

#     plt.legend(ncol=2, loc='lower left')

#     # fig = plt.figure(figsize=(4,2))
#     # plt.errorbar(np.arange(num_outputs), mean_cor_pred_val, yerr=std_cor_pred_val, fmt='o', 
#     #              color=COLOR_DICT['midnight_blue'], ecolor=COLOR_DICT['viz_orange'], elinewidth=3, capsize=4)
#     plt.xlabel('Gesture')
#     plt.xticks(np.arange(num_outputs), class_to_gest.values(), ha='center')
#     plt.ylabel('Accuracy')
#     # plt.ylim([0,1])

#     if wandb_log:
#         wandb.log({f"Last Epoch Accuracy Per Class": wandb.Image(fig)})

#     return fig


# def plot_mean_prec_rec_per_class(num_outputs, mean_prec, std_prec, mean_recall, std_recall, class_to_gest=CLASS_TO_GEST, wandb_log=True):
#     """
#     Plots the mean and std of precision, recall and accuracy per class.
#     Accuracy is the diagonal of the confusion matrix.
    
#     """
#     fig = plt.figure(figsize=(4,2))

#     bwidth= 0.2
#     capsize= 2
    
#     plt.bar(np.arange(num_outputs), mean_prec, yerr=std_prec,
#             width=bwidth, color=COLOR_DICT['naval'],
#             capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='precision')
    
#     print(f"std recall:{std_recall}")
#     plt.bar(np.arange(num_outputs) + bwidth, mean_recall, yerr=std_recall,
#             width=bwidth, color=COLOR_DICT['orange'],
#             capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='recall')
    
  
#     plt.xlabel('Gesture')
#     plt.ylabel('Score')
#     plt.legend(ncol=3, loc='upper left')
#     plt.xticks(np.arange(num_outputs)+bwidth/2, class_to_gest.values(), ha='center', rotation=0)
#     plt.ylim([0,1.2])
#     if wandb_log:
#         wandb.log({f"Last Epoch Precision/Recall Per Class": wandb.Image(fig)})

#     return fig




def plot_mean_firing_rate_temporal(input, labels):

    c = [COLOR_DICT['wisteria'], COLOR_DICT['viz_orange'],  COLOR_DICT['orange'],COLOR_DICT['naval']]
    markers = ['v','o', 's', '^']

    fig = plt.figure(figsize=(10,3))

    # get spike count across time
    sp_rate = input.sum(dim=1).cpu().numpy() / wandb.config['bin_width']  # (n_samples,n_neurons)

    # get mean firing rate per label
    for i, l in enumerate(labels.unique()):
        l_mask = torch.where(labels==l)[0]
        mean_sp_rate = sp_rate[l_mask,:].mean(axis=0)
        std_sp_rate = sp_rate[l_mask,:].std(axis=0)

        plt.plot(np.arange(53), mean_sp_rate[:], markers[i], color=c[i], markersize=5, label=CLASS_TO_GEST[int(l)])

        # add shaded area for std
        plt.fill_between(np.arange(53), mean_sp_rate - std_sp_rate, 
                        mean_sp_rate+std_sp_rate, alpha=0.08, color=c[i])
        
    plt.xlabel('Channels')
    plt.ylabel('Mean Firing Rate [Hz]')
    plt.legend(ncol=4)

    plt.show()



# def plot_acc_weighted_prec_recall(results_df, plot_train=True):
#     fig = plt.figure(figsize=(4,2))
#     ax = fig.add_subplot(111)
#     bwidth=0.2
#     capsize=2
#     shift_bar = 0
#     if plot_train:
#         ax.bar(np.arange(4), results_df[['acc_train','prec_train', 
#                                         'recall_train','f1_train']].mean(axis=0), 
#                                         yerr=results_df[['acc_train','prec_train', 
#                                         'recall_train','f1_train']].std(axis=0),
#                                         width=bwidth, 
#                                         color=COLOR_DICT['wisteria'],capsize=capsize,
#                                         ecolor=COLOR_DICT['midnight_blue'],
#                                         label='train')
#         shift_bar  = bwidth
#     ax.bar(np.arange(4)+ shift_bar, results_df[['acc_val','prec_val', 
#                                     'recall_val','f1_val']].mean(axis=0), 
#                                     yerr=results_df[['acc_val','prec_val', 
#                                     'recall_val','f1_val']].std(axis=0),
#                                     width=bwidth, 
#                                     color=COLOR_DICT['viz_orange'],capsize=capsize,
#                                     ecolor=COLOR_DICT['midnight_blue'],
#                                     label='val')
#     plt.xlabel('Metrics')
#     plt.ylabel('Score')
#     plt.xticks(np.arange(4)+shift_bar/2, ['Accuracy', 'Weighted precison', 'Weighted recall', 'Weighted f1'], ha='center', rotation=0)
#     plt.legend(ncol=2)                            
#     return fig



# #TODO: maybe split the plotting file one for preprocessing and one for other stuffs
# def plot_ch_annotate_trials(eng_dataset, ch, y_range, title, data_type='filt'):
#     if data_type == 'filt':
#         df = eng_dataset.filt_df
#     elif data_type == 'post':
#         df = eng_dataset.post_data_df
#     else:
#         raise ValueError("Data type should be either 'filt' or 'post'")

#     trig_df = eng_dataset.trig_df
#     task_order = eng_dataset.task_order
#     n_tasks  = eng_dataset.n_tasks

#     fig = plt.figure(figsize=(5, 2))
#     ax = fig.add_subplot(111)
#     plot = ax.plot(df[TIME_VAR], df[ch], label= f"Ch {ch}")
#     plt.box(False)

#     for rep_id, trig in enumerate(trig_df[TRIG_VAR]):
#         if rep_id == len(trig_df[TRIG_VAR]) - 1:
#             box_width = df[TIME_VAR].iloc[-1] - trig_df[TRIG_VAR][rep_id]
#         else:
#             box_width = trig_df[TRIG_VAR][rep_id+1] - trig_df[TRIG_VAR][rep_id]
#         facecolor = list(COLOR_DICT.values())[int(trig_df.iloc[rep_id]['task_id'])]
#         box = Rectangle((trig, y_range[0]),box_width, np.abs(y_range[0])+y_range[1], facecolor=facecolor, alpha=0.4,
#                         ) #label=TASK_ORDER[int(trig_df.iloc[rep_id]['task_id'])]
#         ax.axvline(x=trig,ymin= df[ch].min(), ymax= df[ch].max(), color='r', linestyle='--', alpha=0.2)
#         ax.add_patch(box)

#     # add legend
#     legend_patches = []
#     for task in range(n_tasks):
#         legend_patches.append(mpatches.Patch(color=list(COLOR_DICT.values())[task], label=task_order[task], alpha=0.4))
#     plt.legend(handles=legend_patches, ncol=5)
    
#     plt.tight_layout()
#     # plt.legend(handles=box.legend_elements()[0], labels=TASK_ORDER, ncol=5)

#     plt.xlabel('Time [sec]', fontsize=8)
#     plt.ylabel(r'Amplitude [$\mu$V]', fontsize=8)
#     plt.title(title)
#     plt.xlim([0,245])
#     plt.show()

#     if eng_dataset.save_figs:
#         base_filename = f"day{eng_dataset.day}{eng_dataset.session}_{data_type}_ch_{ch}"
#         if data_type=='filt':
#             fill = f"{eng_dataset.filt_pipeline['bp_cutoff_freq'][0]}_{eng_dataset.filt_pipeline['bp_cutoff_freq'][1]}"
#         if data_type=='post':
#             fill = 'raw'
#         fig.savefig(os.path.join(FIG_DIR, f"{base_filename}_{fill}_eng_cut_annotate_reps.png"),
#                      dpi=300, bbox_inches='tight')


