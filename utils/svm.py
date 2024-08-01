import wandb

# from utils.evaluate_model import plot_confusion_matrix, compute_metric, get_perf_results
# from utils.plotting import plot_mean_prec_rec_per_class, plot_mean_acc_per_class, plot_acc_weighted_prec_recall
# from utils.preprocess_analog import select_channels, extract_feat_for_task, compute_stat_in_time_all, extract_temporal_feat
import utils.plot as uplot
from constants import *

from srcs.engdataset import ENGDataset
import utils.preprocessing as pre

import random
import numpy as np
import pandas as pd
from collections import Counter
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
import logging
import dataframe_image as dfi
import matplotlib.pyplot as plt
import itertools

LABELS_SHORT = {0: "pinch", 1: "opp", 2: "power", 3: "ulnar", 4: "abd"}
PERF_METRICS = ['acc_train', 'acc_val', 'f1_train', 'f1_val', 'prec_train', 'prec_val', 'recall_train', 'recall_val']


def filter_dict(d, label_list):
    """
    Filter dict by list of keys (i.e list of class labels)
    """
    return {l: d[l] for l in label_list}



def train_svm(feat_df, id_to_label_short, select_class, chs_to_exclude=[1,2,43],k_cv=3, save_fig=True,annotate_cm=False, shuffle=True, 
              is_temporal=False, rep_list=None, seed=None, return_conf_test=False, bin_width=1, bin_stat='pow',exp_var=None, kernel='linear',**kwargs):
    """
    feat_df: dataframe with the features and all the labels
    id_to_label_short: dict with the labels and the corresponding gesture names. Key: label, value: gesture name
    select_class: list of labels to use for the classification. This is used because sometime I don't want to classify all the labels in feat_df but a subset of them.

    rep_list: list of rep_ids. This is used in the case of is_temporal=True. Since each rep is unfolded into small time bins, this list is used to map each time bin to a rep id.
    bin_width: in case of temporal data, the bin width is a number between 0 and 1
    """
    assert len(id_to_label_short) == len(np.unique(feat_df['label'])),  "labels dict and feat_df['label'] do not match"

    random.seed(seed)
    np.random.seed(seed)
    # select from dict of class names
    label_to_gest = filter_dict(id_to_label_short, select_class)
    print(f"\nlabels:{id_to_label_short}  \nlabels_to_gest:{label_to_gest} \nfeat_df_label:{feat_df['label'].unique()}")

    
    # Select labels to use
    if not is_temporal: # filter irrelevant labels
        clf_df = feat_df[feat_df['label'].isin(select_class)]
    else: # in case of temporal, we keep all labels to have the global rep_id which is used later to split the data into train and test sets
        clf_df = feat_df
    print("clf_df['label'].unique():", clf_df['label'].unique())

    

    # define dataset
    X = clf_df.iloc[:, :-1]  # in case of temporal: last two columns are rep_id and label. We still need the rep_id to split the data into train and test sets
    y = clf_df.iloc[:, -1]
    

    # remove channels to exclude
    X = select_channels(X, chs_to_exclude)
    

    print(X.shape, y.shape)

    # Split data cross-validation
    kf = StratifiedKFold(n_splits=k_cv,shuffle=shuffle, random_state=seed)

    if is_temporal:
        print(f"is temporal: (len(rep_list):{len(rep_list)})")
        sel_gest_rep_list = [i for i in rep_list if i in select_class]
        sel_rep_id_list = [index for index,j in enumerate(rep_list) if j  in select_class]
        print(f"sel_gest_rep_list ({len(sel_gest_rep_list)}):{sel_gest_rep_list}")
        print(f"sel_rep_id_list  ({len(sel_rep_id_list)}):{sel_rep_id_list}")

        # rep_gest_id = np.arange(len(rep_list))
        # kf_split = kf.split(np.arange(len(rep_list)), rep_list)  # split over the ids of the entire dataset
        kf_split = kf.split(sel_rep_id_list, sel_gest_rep_list)
        
    else:
        kf_split = kf.split(X,y)



    results_dict = {'acc_val': [], 'acc_train': [], 'f1_val': [], 'f1_train': [], 'conf_matrix_train': [],
                    'conf_matrix_val': [], 'class_rep_train': [], 'class_rep_val': [],'prec_val':[],
                    'prec_train':[], 'recall_val':[], 'recall_train':[]}



    num_classes = len(select_class)
    conf_matrix_fold_val = np.empty((k_cv, num_classes, num_classes))
    prec_class_fold_val = np.empty((k_cv, num_classes))
    recall_class_fold_val = np.empty((k_cv, num_classes))

    conf_matrix_fold_train = np.empty((k_cv, num_classes, num_classes))

    for fold_i, (train_ind, test_ind) in enumerate(kf_split):
        if is_temporal:
            print(f"Fold:{fold_i}    RELATIVE  Train_ind:{train_ind}   Test_ind:{test_ind}")
            # get global train and test indices
            train_ind = [sel_rep_id_list[i] for i in train_ind]
            test_ind = [sel_rep_id_list[i] for i in test_ind]

            print(f"Fold:{fold_i}    global  Train_ind:{train_ind}   Test_ind:{test_ind}\n----------------------\n")
            X_train = X[X['rep_id'].isin(train_ind)].drop(['rep_id'], axis=1)
            X_test = X[X['rep_id'].isin(test_ind)].drop(['rep_id'], axis=1)

            y_train = y[X['rep_id'].isin(train_ind)]
            y_test = y[X['rep_id'].isin(test_ind)]

            print(f"unqiue y_test:{np.unique(y_test)}  y_train:{np.unique(y_train)}  y:{np.unique(y)}")


        else:
            X_train = X.iloc[train_ind, :]
            y_train = y.iloc[train_ind]
            
            X_test = X.iloc[test_ind, :]
            y_test = y.iloc[test_ind]

        print(f"Fold {fold_i}: {Counter(y_train)} , {Counter(y_test)}")
        print(f"Train: {X_train.shape} , {y_train.shape} Test: {X_test.shape} , {y_test.shape}\n-------------------")

        # define model
        clf = OneVsRestClassifier(SVC(kernel=kernel, probability=True,**kwargs))

        # fit model
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_test)
        # y_pred_train = clf.predict(X_train)

        logging.info(f"Fold :{fold_i} pred_labels: {np.unique(y_pred)}")

        # evaluate predictions
        results_dict = get_perf_results(clf, results_dict, X_train, y_train, X_test, y_test)
        conf_matrix_fold_val[fold_i, :, :] = results_dict['conf_matrix_val'][-1]
        conf_matrix_fold_train[fold_i, :, :] = results_dict['conf_matrix_train'][-1]

        prec, recall, _ ,_= compute_metric(y_test, y_pred,'precision_recall_fscore_support',
                                           CLASS_TO_GEST, avg=None)
        prec_class_fold_val[fold_i, :] = prec   
        recall_class_fold_val[fold_i, :] = recall

        # plot confusion matrix
        uplot.plot_confusion_matrix(y_test, y_pred,CLASS_TO_GEST,annotate=annotate_cm)
    results_df = pd.DataFrame(results_dict)[PERF_METRICS]
    print(f"Results_df:{results_df}\n ")
    print(f"Mean acc train:{results_df['acc_train'].mean()} + {results_df['acc_train'].std()}\nMean acc val:{results_df['acc_val'].mean()} + {results_df['acc_val'].std()}\n----------------")


    # Mean and std of confusion matrix across folds
    mean_conf_matrix_test = get_mean_across_folds( conf_matrix_fold_val) #np.mean(conf_matrix_fold_val, axis=0) 
    std_conf_matrix_test = get_sd_across_folds(conf_matrix_fold_val) #np.std(conf_matrix_fold_val, axis=0)

    mean_conf_matrix_train =  get_mean_across_folds(conf_matrix_fold_train)  #np.mean(conf_matrix_fold_train, axis=0) 
    std_conf_matrix_train = get_sd_across_folds(conf_matrix_fold_train)  #np.std(conf_matrix_fold_train, axis=0)

    # Mean and std of precision and recall across folds
    mean_prec_class = np.mean(prec_class_fold_val, axis=0)
    std_prec_class = np.std(prec_class_fold_val, axis=0)

    mean_recall = np.mean(recall_class_fold_val, axis=0)
    std_recall = np.std(recall_class_fold_val, axis=0)

    # Plot Results
    print(f"Mean prec:{mean_prec_class}  std prec:{std_prec_class}  \nMean recall:{mean_recall}  std recall:{std_recall}\n")
    fig_prec = uplot.plot_mean_prec_rec_per_class(len(select_class), mean_prec_class, std_prec_class, mean_recall, std_recall, wandb_log=False)
    wandb.log({f"Precision/Recall per Class": wandb.Image(fig_prec)})

    # plot confusion matrix
    fig = uplot.plot_confusion_matrix(y_test, y_pred, CLASS_TO_GEST, title=f"Mean predictions across {k_cv} folds", matrix=mean_conf_matrix_test, annotate=annotate_cm)
    wandb.log({f"Mean Confusion Matrix": wandb.Image(fig)})
    mean_acc = np.diag(mean_conf_matrix_test).mean()
    wandb.log({f"mean_corr_pred": mean_acc})


    # plot correct predictions: diagonal of confusion matrix
    fig_mean_acc = uplot.plot_mean_acc_per_class(len(select_class), mean_conf_matrix_test, std_conf_matrix_test, 
                                             mean_conf_matrix_train, std_conf_matrix_train,
                                           wandb_log=False)
    wandb.log({f"Mean Accuracy Per Class": wandb.Image(fig_mean_acc)})


    fig_weighted_prec = uplot.plot_acc_weighted_prec_recall(results_df)
    wandb.log({f"Weighted Precision/Recall": wandb.Image(fig_weighted_prec)})


    # save figure for confusion matrix and results df
    if exp_var is not None:
        filename_prefix= f"seed_{seed}_{bin_stat}_kcv_{k_cv}_nchs_{X.shape[1]}_expvar_{exp_var:.1f}"
    else:
        filename_prefix= f"seed_{seed}_{bin_stat}_kcv_{k_cv}"

    if is_temporal:
        if wandb.config['use_mov_avg']:
            filename_suffix = f"{list(label_to_gest.values())}_temporal_{is_temporal}_wind_{bin_width}_overlap_{wandb.config['overlap_perc']}"

        else:
            filename_suffix = f"{list(label_to_gest.values())}_temporal_{is_temporal}_bin_{bin_width}"
    else:
        filename_suffix = f"{list(label_to_gest.values())}"

    conf_fig_file = f"{filename_prefix}_conf_matrix_{kernel}_{filename_suffix}.png"
    results_df_file = f"{filename_prefix}_acc_df_{kernel}_{filename_suffix}.png"
    prec_fig_file = f"{filename_prefix}_prec_recall_per_class_{kernel}_{filename_suffix}.png"
    acc_fig_file = f"{filename_prefix}_mean_acc_per_class_{kernel}_{filename_suffix}.png"
    weighted_fig_file = f"{filename_prefix}_overall_weighted_prec_recall_{kernel}_{filename_suffix}.png"

    if save_fig:
        logging.info(f"Saving figure to {os.path.join(CLF_FIG,DAY_SESS_FIG,conf_fig_file)}\n{os.path.join(CLF_FIG, results_df_file)}")
        fig.savefig(os.path.join(CLF_FIG, DAY_SESS_FIG,conf_fig_file), dpi=300, bbox_inches='tight')
        fig_prec.savefig(os.path.join(CLF_FIG, DAY_SESS_FIG,prec_fig_file), dpi=300, bbox_inches='tight')
        fig_mean_acc.savefig(os.path.join(CLF_FIG, DAY_SESS_FIG, acc_fig_file), dpi=300, bbox_inches='tight')
        fig_weighted_prec.savefig(os.path.join(CLF_FIG,DAY_SESS_FIG, weighted_fig_file), dpi=300, bbox_inches='tight')
        
        dfi.export(results_df, os.path.join(CLF_FIG, DAY_SESS_FIG,results_df_file))
        

    if return_conf_test:
        return results_df,mean_conf_matrix_test,std_conf_matrix_test
    
    return results_df





# Prepare dataset for training
def prepare_input_df(eng_dataset:ENGDataset, feature:str, organize_strat:str,  wind_size:float, overlap_perc:float):
    #TODO: add the non-temporal version where the feature is computed across the whole duration of each repetition
    """
    organize_strat (str): defines the classes we would like to discriminate. Can be 'flx_vs_ext_combined' or 'flx_vs_ext_separate' or 'flx_vs_ext_vs_rest'. 
        - 'flx_vs_ext_combined': combine flexions of all tasks into one class and extensions of all tasks into another class. In total: 2 classes + rest
        - 'flx_vs_ext_separate': considers each flexion and extension phases of each gesture as 2 distince classes. Since we have 5 tasks, we have 10 classes + rest in total if you select this method.
        - 'flx_and_ext_separate': considers flexion and extension phases of each gesture as a single class. For the 5 tasks, we would then have 5 classes + rest in total.
    
    """
    input_df = pd.DataFrame()
    for task_id in range(len(eng_dataset.task_order)):
        for rep_id in range(eng_dataset.task_rep_count[task_id]):
            flex_feat_df, ext_feat_df, rest_feat_df = pre.extract_feature_for_task_rep(eng_dataset, task_id, rep_id, feature,
                                                                                      wind_size, overlap_perc=overlap_perc)

            # # transpose
            # flex_df = flex_df.T
            # ext_df = ext_df.T
            # rest_df = rest_df.T

            # add labels column
            if organize_strat == 'flx_vs_ext_combined':  
                rest_label = 2
                flex_label = 0
                ext_label = 1
                labels_map = {flex_label: 'Close', ext_label: 'Open', rest_label: 'Rest'}
                flex_feat_df[LABEL_COL] = [flex_label for rep in range(eng_dataset.task_rep_count[task_id])]
                ext_feat_df[LABEL_COL] = [ext_label for rep in range(eng_dataset.task_rep_count[task_id])]
     
            elif organize_strat == 'flx_and_ext_separate': 
                flex_feat_df[LABEL_COL] = task_id #[task_id for rep in range(eng_dataset.task_rep_count[task_id])]
                ext_feat_df[LABEL_COL] = task_id #[task_id for rep in range(eng_dataset.task_rep_count[task_id])]
                rest_label = 5
                labels_map = {0: "Tripod", 1: "ThOpp", 2: "Power", 3: "UlnarFing.", 4: "FingAbd.", rest_label: "Rest"}

            #TODO: refactor the other labels_map to be consistent with this one
            elif organize_strat == 'flx_vs_ext_separate': 
                flex_feat_df[LABEL_COL] = task_id*2 #[task_id*2 for rep in range(eng_dataset.task_rep_count[task_id])]        
                ext_feat_df[LABEL_COL] = task_id*2+1 #[task_id*2+1 for rep in range(eng_dataset.task_rep_count[task_id])]

                rest_label = 5 * 2
                labels_map = {0: f"{eng_dataset.task_order[0]} Close", 1: f"{eng_dataset.task_order[0]} Open", 
                              2: f"{eng_dataset.task_order[1]} Close", 3: f"{eng_dataset.task_order[1]} Open", 
                              4: f"{eng_dataset.task_order[2]} Close", 5: f"{eng_dataset.task_order[2]} Open", 
                              6: f"{eng_dataset.task_order[3]} Close", 7: f"{eng_dataset.task_order[3]} Open", 
                              8: f"{eng_dataset.task_order[4]} Close", 9: f"{eng_dataset.task_order[4]} Open", 
                              rest_label: "Rest"}
            rest_feat_df[LABEL_COL] = rest_label #[rest_label for rep in range(eng_dataset.task_rep_count[task_id])]
  
            # concat
            input_df = pd.concat([input_df, flex_feat_df, ext_feat_df, rest_feat_df], axis=0)
    return input_df, labels_map



def create_feat_df(df, trig_df, clf_classes, reps_count_dict, feat_list=['pow']):
    """
    Creates a dataframe with the features and the labels. 
    clf_classes (string): defines the classes we would like to discriminate. Can be 'flx_vs_ext_combined' or 'flx_vs_ext_separate' or 'flx_vs_ext_vs_rest'. 

        - 'flx_vs_ext_combined': combine flexions of all tasks into one class and extensions of all tasks into another class. In total: 2 classes + rest
        - 'flx_vs_ext_separate': considers each flexion and extension phases of each gesture as 2 distince classes. Since we have 5 tasks, we have 10 classes + rest in total if you select this method.
        - 'flx_and_ext_separate': considers flexion and extension phases of each gesture as a single class. For the 5 tasks, we would then have 5 classes + rest in total.
    """
    
    feat_df = pd.DataFrame()
    for task_id in range(N_TASKS):
        for feat in feat_list:
            flex_df, ext_df, rest_df = extract_feat_for_task(df, task_id, trig_df, reps_count_dict, FLX_DUR, EXT_DUR, REST_DUR, feat=feat)

            # transpose
            flex_df = flex_df.T
            ext_df = ext_df.T
            rest_df = rest_df.T

            # add labels
            if clf_classes == 'flx_vs_ext_combined':  
                REST_LABEL = 2
                FLX_LABEL = 0
                EXT_LABEL = 1
                LABELS = {FLX_LABEL: 'Close', EXT_LABEL: 'Open', REST_LABEL: 'Rest'}
                flex_df['label'] = [FLX_LABEL for rep in range(reps_count_dict[task_id])]
                ext_df['label'] = [EXT_LABEL for rep in range(reps_count_dict[task_id])]
     
            elif clf_classes == 'flx_and_ext_separate': 
                flex_df['label'] = [task_id for rep in range(reps_count_dict[task_id])]
                ext_df['label'] = [task_id for rep in range(reps_count_dict[task_id])]
                REST_LABEL = 5
                LABELS = {0: "Tripod", 1: "ThOpp", 2: "Power", 3: "UlnarFing.", 4: "FingAbd.", REST_LABEL: "Rest"}

        
            elif clf_classes == 'flx_vs_ext_separate': 
                flex_df['label'] = [task_id*2 for rep in range(reps_count_dict[task_id])]
                ext_df['label'] = [task_id*2+1 for rep in range(reps_count_dict[task_id])]
                REST_LABEL = 5 * 2
                LABELS = {0: "Tripod Close", 1:"Tripod Open", 
                          2: "ThOpp Close", 3: "ThOpp Open", 
                          4: "Power Close", 5: "Power Open", 
                          6: "UlnarFing. Close", 7:"UlnarFing. Open", 
                          8: "FingAbd. Close ", 9:"FingAdd. Open", 
                          REST_LABEL: "Rest"}
            rest_df['label'] = [REST_LABEL for rep in range(reps_count_dict[task_id])]

            # concat
            feat_df = pd.concat([feat_df, flex_df, ext_df, rest_df], axis=0)
    return feat_df, LABELS





def train_in_time(filt_df, trig_df_post, n_reps, sel_gest, bin_width, bin_stat, select_class= [0,1,3,4], 
                  plot_feat=False, seed=42, temp_feat_df=None, k_cv=3, save_fig=False, 
                  kernel='linear', **kwargs):

    logging.info(f"Computing {bin_stat} over time for {sel_gest}")

    # wind_size= bin_width #wandb.config['wind_size']  
    stride = bin_width  - wandb.config['overlap_perc'] * bin_width    # duration in sec
    min_periods= 0 #wind_size

    print(f"wind_size:{bin_width}  stride:{stride}  min_periods:{min_periods}")

    if temp_feat_df is None: # used for in case of debugging mode to avoid recomputing
        if wandb.config['use_mov_avg']:
            #TODO: change function call; it was refactored
            temp_feat_df_mv = extract_temporal_feat(filt_df, trig_df_post, n_reps, sel_gest, bin_width, stride, min_periods, feat='pow')
            temp_feat_df = temp_feat_df_mv.drop([TIME_VAR], axis=1)
        else:
            temp_feat_df = compute_state_in_time_all(filt_df, trig_df_post, n_reps, sel_gest, FLX_DUR, EXT_DUR, REST_DUR, bin_width, bin_stat)
            temp_feat_df = temp_feat_df.drop([TIME_VAR], axis=1)

    # train svm on those bins
    print(f"feat_df shape:{temp_feat_df.shape}")

    # train svm on those bins
    rep_list  = list(itertools.chain(*[[sel_gest_item.id]*n_reps[sel_gest_item.id]  for sel_gest_item in sel_gest]))
    gest_id_to_label = {i.id:f'{LABELS_SHORT[i.id]} {i.phase}' for i in sel_gest}
    logging.info(f'select_class: {select_class}   labels:{gest_id_to_label} sel_gest: {sel_gest}')

    results_df, conf_test_mean, conf_test_sd = train_svm(temp_feat_df, gest_id_to_label, select_class, save_fig=save_fig, annotate_cm=False, shuffle=True, is_temporal=True, rep_list= rep_list, seed=seed, return_conf_test=True,
                                      k_cv=k_cv, bin_width=bin_width, kernel=kernel, **kwargs)

    return results_df,conf_test_mean, conf_test_sd




def create_figures_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    if not os.path.exists(CLF_FIG):
        os.makedirs(CLF_FIG)



def get_mean_across_folds(matrix):
    return  np.mean(matrix, axis=0)

def get_sd_across_folds(matrix):
    return  np.std(matrix, axis=0)
