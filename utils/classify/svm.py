from srcs.engdataset import ENGDataset
from utils.preprocessing.extract_feat import extract_feature_for_task_rep_per_phase
from utils.plot.model_perf import *
from utils.evaluate_model import get_perf_results, compute_metric
from constants import *


from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from collections import Counter
import logging
import random
import pandas as pd
from typing import List, Dict
import dataframe_image as dfi


# Prepare dataset for training
def prepare_input_df(eng_dataset: ENGDataset, feature: str, organize_strat: str,  wind_size: float, 
                     overlap_perc: float, over_entire_rep:bool=False):
    # TODO: add the non-temporal version where the feature is computed across the whole duration of each repetition
    """
    organize_strat (str): defines the classes we would like to discriminate. Can be 'flx_vs_ext_combined' or 'flx_vs_ext_separate' or 'flx_vs_ext_vs_rest'. 
        - 'flx_vs_ext_combined': combine flexions of all tasks into one class and extensions of all tasks into another class. In total: 2 classes + rest
        - 'flx_vs_ext_separate': considers each flexion and extension phases of each gesture as 2 distince classes. Since we have 5 tasks, we have 10 classes + rest in total if you select this method.
        - 'flx_and_ext_separate': considers flexion and extension phases of each gesture as a single class. For the 5 tasks, we would then have 5 classes + rest in total.
    
    """
    input_df = pd.DataFrame()
    num_tasks = len(eng_dataset.task_order)
    num_phases = 2     # flexion and extension or Close and Open
    for task_id in range(num_tasks):
        for rep_id in range(eng_dataset.task_rep_count[task_id]):
            flex_feat_df, ext_feat_df, rest_feat_df = extract_feature_for_task_rep_per_phase(eng_dataset,
                                                                                   task_id,
                                                                                   rep_id,
                                                                                   feature,
                                                                                   wind_size,
                                                                                   overlap_perc=overlap_perc,
                                                                                   over_entire_rep=over_entire_rep)

            # add labels column
            if organize_strat == 'flx_vs_ext_combined':
                rest_label = 2
                flex_label = 0
                ext_label = 1
                labels_map = {flex_label: 'Close',
                              ext_label: 'Open', 
                              rest_label: 'Rest'}
                
                flex_feat_df[LABEL_COL] = [flex_label for rep in range(eng_dataset.task_rep_count[task_id])]
                ext_feat_df[LABEL_COL] = [ext_label for rep in range(eng_dataset.task_rep_count[task_id])]

            elif organize_strat == 'flx_and_ext_separate':
                # [task_id for rep in range(eng_dataset.task_rep_count[task_id])]
                flex_feat_df[LABEL_COL] = task_id
                # [task_id for rep in range(eng_dataset.task_rep_count[task_id])]
                ext_feat_df[LABEL_COL] = task_id
                rest_label = 5
                labels_map = {0: "Tripod", 1: "ThOpp", 2: "Power",
                              3: "UlnarFing.", 4: "FingAbd.", rest_label: "Rest"}

            # TODO: refactor the other labels_map to be consistent with this one
            elif organize_strat == 'flx_vs_ext_separate':

                flex_feat_df[LABEL_COL] = task_id * num_phases
                ext_feat_df[LABEL_COL] = task_id * num_phases +1

                rest_label = num_tasks * 2
                labels_map = {0: f"{eng_dataset.task_order[0]} Close", 1: f"{eng_dataset.task_order[0]} Open",
                              2: f"{eng_dataset.task_order[1]} Close", 3: f"{eng_dataset.task_order[1]} Open",
                              4: f"{eng_dataset.task_order[2]} Close", 5: f"{eng_dataset.task_order[2]} Open",
                              6: f"{eng_dataset.task_order[3]} Close", 7: f"{eng_dataset.task_order[3]} Open",
                              8: f"{eng_dataset.task_order[4]}", 9: f"FingAdd.",
                              rest_label: "Rest"}

            rest_feat_df[LABEL_COL] = rest_label

            # concatenate the dataframes
            input_df = pd.concat([input_df, flex_feat_df, ext_feat_df, rest_feat_df], axis=0)
    return input_df, labels_map



def filter_dict(d, label_list):
    """
    Filter dict by list of keys (i.e list of class labels)
    """
    return {l: d[l] for l in label_list}



def fit_svm(input_df, labels_map: Dict[int, str], select_class: List[int],
            eng_dataset:ENGDataset,
             bad_channels:List[int] =[],
              k_cv=3, save_fig=True, annotate_cm=False, shuffle=True, is_temporal=False, 
              seed=None, return_conf_test=False,bin_width=None, bin_stat=None, exp_var=None,
              kernel='linear'):
    """
    feat_df: dataframe with the features and all the labels
    labels_map: dict with the labels and the corresponding gesture names. Key: label, value: gesture name
    select_class: list of labels to use for the classification. This is used because sometime I don't want to classify all the labels in feat_df but a subset of them.

    rep_list: list of rep_ids. This is used in the case of is_temporal=True. Since each rep is unfolded into small time bins, this list is used to map each time bin to a rep id.
    bin_width: in case of temporal data, the bin width is a number between 0 and 1
    """
    metrics = ['acc', 'f1', 'prec', 'recall']
    datasets = ['train', 'val']
    random.seed(seed)
    np.random.seed(seed)

    assert len(labels_map) == input_df[LABEL_COL].nunique(),  "labels map and feat_df['label'] do not match"

    if is_temporal:
        assert bin_width is not None, "bin_width must be provided for temporal data"
        assert bin_stat is not None, "bin_stat must be provided for temporal data"


    # select from lavels_map the classes we want to consider for classification
    selected_labels_map = filter_dict(labels_map, select_class)

    clf_df = input_df[input_df[LABEL_COL].isin(select_class)]

    # define dataset
    aux_cols = [TIME_VAR, REP_ID_COL, LABEL_COL, FEAT_WIN_COL]

    # remove channels to exclude
    if len(bad_channels) > 0:
        clf_df = clf_df[[col for col in clf_df.columns if col not in bad_channels]]

    # Split data cross-validation
    kf = StratifiedKFold(n_splits=k_cv, shuffle=shuffle, random_state=seed)

    if is_temporal:
        temp_df = clf_df.groupby([LABEL_COL, REP_ID_COL], as_index=False)[FEAT_WIN_COL].count()

        # create a map from each class_id and rep_id into a global id: identifying how many samples in total are used for classification
        # with disregard for the time window (i.e ignoring the splitting of each sample into time windows)
        temp_df['global_sample_id'] = temp_df.index

        # merge the global id to the input_df
        clf_df = pd.merge(clf_df, temp_df[[LABEL_COL, REP_ID_COL, 'global_sample_id']], on=[
                          LABEL_COL, REP_ID_COL], how='left')
        glob_target_class = clf_df.groupby([LABEL_COL, REP_ID_COL, 'global_sample_id'], as_index=False)[
            FEAT_WIN_COL].count()[LABEL_COL].to_numpy()

        logging.info(f"clf_df['label'].unique():{clf_df[LABEL_COL].unique()}\n")

        # splitting with stratefied on the global target class (n_reps * n_classes)
        kf_split = kf.split(glob_target_class, glob_target_class)

    else:
        # strip aux columns
        X = clf_df.drop(aux_cols, axis=1)
        y = clf_df[LABEL_COL]
        print(X.shape, y.shape)

        kf_split = kf.split(X, y)



    results_dict = {f'{metric}_{dataset}': [] for metric in metrics for dataset in datasets}
    results_dict.update({f'conf_matrix_{dataset}': [] for dataset in datasets})
    results_dict.update({f'class_rep_{dataset}': [] for dataset in datasets})

    perf_metrics = [f'{metric}_{dataset}' for metric in metrics for dataset in datasets]

    conf_matrix_fold = np.empty((k_cv, len(select_class), len(select_class)))
    prec_class_fold = np.empty((k_cv, len(select_class)))
    recall_class_fold = np.empty((k_cv, len(select_class)))

    for fold_i, (train_ind, test_ind) in enumerate(kf_split):
        if is_temporal:
          
            X_train = clf_df[clf_df['global_sample_id'].isin(train_ind)].drop(
                aux_cols + ['global_sample_id'], axis=1)
            X_test = clf_df[clf_df['global_sample_id'].isin(test_ind)].drop(
                aux_cols + ['global_sample_id'], axis=1)

            y_train = clf_df[clf_df['global_sample_id'].isin(train_ind)][LABEL_COL]
            y_test = clf_df[clf_df['global_sample_id'].isin( test_ind)][LABEL_COL]
            
        else:
            X_train = X.iloc[train_ind, :]
            y_train = y.iloc[train_ind]

            X_test = X.iloc[test_ind, :]
            y_test = y.iloc[test_ind]

     
        # define model
        clf = OneVsRestClassifier(SVC(kernel=kernel, probability=True))

        # fit model
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_test)
       

        logging.info(f"Fold {fold_i}\n-------------\nTrain sample ids:{train_ind}\nTest sample ids:{test_ind}\nCounter train:{Counter(y_train)}\nCounter test: {Counter(y_test)} \npred_labels: {np.unique(y_pred)}")
        logging.info(f"Train: {X_train.shape} , {y_train.shape} Test: {X_test.shape} , {y_test.shape}\n-------------------\n")

        # evaluate predictions
        results_dict = get_perf_results( clf, results_dict, X_train, y_train, X_test, y_test, datasets)
        conf_matrix_fold[fold_i, :, :] = results_dict['conf_matrix_val'][-1]

        prec, recall, _, _ = compute_metric(y_test, y_pred, 'precision_recall_fscore_support',
                                            selected_labels_map, avg=None)
        prec_class_fold[fold_i, :] = prec
        recall_class_fold[fold_i, :] = recall

        # plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, selected_labels_map, annotate=annotate_cm)

    results_df = pd.DataFrame(results_dict)[perf_metrics]

    # Mean and std of confusion matrix across folds
    mean_conf_matrix_test = np.mean(conf_matrix_fold, axis=0)
    std_conf_matrix_test = np.std(conf_matrix_fold, axis=0)

    # Mean and std of precision and recall across folds
    mean_prec_class = np.mean(prec_class_fold, axis=0)
    std_prec_class = np.std(prec_class_fold, axis=0)

    mean_recall = np.mean(recall_class_fold, axis=0)
    std_recall = np.std(recall_class_fold, axis=0)
    print(f"\n\n recall class fold:{recall_class_fold}")
    print(
        f"Mean prec:{mean_prec_class}  std prec:{std_prec_class}  Mean recall:{mean_recall}  std recall:{std_recall}")

    # For saving figures and results_df
    filename_prefix = None
    filename_suffix = None
    if save_fig:
        n_channels = X_train.shape[1]
        if exp_var is not None:
            filename_prefix = f"seed_{seed}_{bin_stat}_kcv_{k_cv}_nchs_{n_channels}_expvar_{exp_var}"
        else:
            filename_prefix = f"seed_{seed}_{bin_stat}_kcv_{k_cv}_nchs_{n_channels}"

        filename_suffix = f"{list(selected_labels_map.values())}_temporal_{is_temporal}_bin_{bin_width}"

    # TODO fix function to take any class to gest
    num_test_classes = len(np.unique(y_test))
    plot_mean_prec_rec_per_class(num_test_classes,
                                       mean_prec_class,
                                       std_prec_class,
                                       mean_recall,
                                       std_recall,
                                       selected_labels_map,
                                       wandb_log=False,
                                       save_fig=save_fig,
                                       filename_prefix=filename_prefix,
                                       filename_suffix=filename_suffix)

    # plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, 
                                selected_labels_map,
                                title=f"Mean predictions across {k_cv} folds",
                                matrix=mean_conf_matrix_test, 
                                annotate=annotate_cm,
                                save_fig=save_fig,
                                filename_prefix=filename_prefix,
                                filename_suffix=filename_suffix)

    # save results_df as a png
    results_df_file = f"{filename_prefix}_acc_df_linear_{filename_suffix}.png"
    dfi.export(results_df, os.path.join(CLF_FIG,  results_df_file))
    logging.info(f"Results_df as a fig is saved to {os.path.join(CLF_FIG,  results_df_file)}")

    # save results_df as pkl
    results_df_file = f"svm_{eng_dataset.day}{eng_dataset.session}_{filename_prefix}_acc_df_linear_{filename_suffix}.pkl"
    results_df.to_pickle(os.path.join(CLF_RESULTS_DIR, results_df_file))
    if return_conf_test:
        return results_df, mean_conf_matrix_test, std_conf_matrix_test

    return results_df