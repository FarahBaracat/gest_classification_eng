from constants import *

import matplotlib.pyplot as plt
import numpy as np
import wandb
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

import logging

def plot_mean_acc_per_class(num_outputs, mean_conf_matrix_val, std_conf_matrix_val, 
                            mean_conf_matrix_train=None,std_conf_matrix_train=None, 
                            class_to_gest=CLASS_TO_GEST, wandb_log=True):
    
    # figure styling
    bwidth= 0.2
    capsize= 2

    fig = plt.figure(figsize=(4,2))
    mean_cor_pred_val = np.diag(mean_conf_matrix_val) # average is across the folds
    std_cor_pred_val = np.diag(std_conf_matrix_val)  # sd is across the folds

    print(f"Mean corr pred val: {mean_cor_pred_val}\n std corr pred val: {std_cor_pred_val}")

    plt.bar(np.arange(num_outputs) + bwidth, mean_cor_pred_val, yerr=std_cor_pred_val,
                width=bwidth, color=COLOR_DICT['viz_orange'],
                capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='val')
    
    if mean_conf_matrix_train is not None: # used for plotting the train accuracy if provided, this is the case with the SVM analysis
        mean_cor_pred_train = np.diag(mean_conf_matrix_train) 
        std_cor_pred_train = np.diag(std_conf_matrix_train)  
        print(f"Mean corr pred train: {mean_cor_pred_train}\n std corr pred train: {std_cor_pred_train}")

        plt.bar(np.arange(num_outputs), mean_cor_pred_train, yerr=std_cor_pred_train,
            width=bwidth, color=COLOR_DICT['wisteria'],
            capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='train')

    plt.legend(ncol=2, loc='lower left')

    # fig = plt.figure(figsize=(4,2))
    # plt.errorbar(np.arange(num_outputs), mean_cor_pred_val, yerr=std_cor_pred_val, fmt='o', 
    #              color=COLOR_DICT['midnight_blue'], ecolor=COLOR_DICT['viz_orange'], elinewidth=3, capsize=4)
    plt.xlabel('Gesture')
    plt.xticks(np.arange(num_outputs), class_to_gest.values(), ha='center')
    plt.ylabel('Accuracy')
    # plt.ylim([0,1])

    # if wandb_log:
    #     wandb.log({f"Last Epoch Accuracy Per Class": wandb.Image(fig)})

    return fig



def plot_mean_prec_rec_per_class(num_outputs, mean_prec, std_prec, mean_recall, std_recall, 
                                 class_to_gest=CLASS_TO_GEST, wandb_log=False, save_fig =False,
                                 filename_prefix='',filename_suffix=''):
    """
    Plots the mean and std of precision, recall and accuracy per class.
    Accuracy is the diagonal of the confusion matrix.
    
    """
    # figure styling
    bwidth= 0.2
    capsize= 2

    fig = plt.figure(figsize=(6,3))

    plt.bar(np.arange(num_outputs), mean_prec, yerr=std_prec,
            width=bwidth, color=COLOR_DICT['naval'],
            capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='precision')
    
    print(f"std recall:{std_recall}")
    plt.bar(np.arange(num_outputs) + bwidth, mean_recall, yerr=std_recall,
            width=bwidth, color=COLOR_DICT['orange'],
            capsize=capsize, ecolor=COLOR_DICT['midnight_blue'], label='recall')
    
  
    plt.xlabel('Gesture')
    plt.ylabel('Score')
    plt.legend(ncol=3, loc='upper left')
    plt.xticks(np.arange(num_outputs)+bwidth/2, class_to_gest.values(), ha='center', rotation=0)
    plt.ylim([0,1.2])
    # plt.show()
    # if wandb_log:
    #     wandb.log({f"Last Epoch Precision/Recall Per Class": wandb.Image(fig)})
    if save_fig:
        prec_fig_file = f"{filename_prefix}_prec_recall_per_class_linear_{filename_suffix}.png"
        fig.savefig(os.path.join(CLF_FIG, prec_fig_file), dpi=300, bbox_inches='tight')



def plot_acc_weighted_prec_recall(results_df, plot_train=True):
    
    # figure styling
    bwidth= 0.2
    capsize= 2

    fig = plt.figure(figsize=(4,2))
    ax = fig.add_subplot(111)

    shift_bar = 0
    if plot_train:
        ax.bar(np.arange(4), results_df[['acc_train','prec_train', 
                                        'recall_train','f1_train']].mean(axis=0), 
                                        yerr=results_df[['acc_train','prec_train', 
                                        'recall_train','f1_train']].std(axis=0),
                                        width=bwidth, 
                                        color=COLOR_DICT['wisteria'],capsize=capsize,
                                        ecolor=COLOR_DICT['midnight_blue'],
                                        label='train')
        shift_bar  = bwidth
    ax.bar(np.arange(4)+ shift_bar, results_df[['acc_val','prec_val', 
                                    'recall_val','f1_val']].mean(axis=0), 
                                    yerr=results_df[['acc_val','prec_val', 
                                    'recall_val','f1_val']].std(axis=0),
                                    width=bwidth, 
                                    color=COLOR_DICT['viz_orange'],capsize=capsize,
                                    ecolor=COLOR_DICT['midnight_blue'],
                                    label='val')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(np.arange(4)+shift_bar/2, ['Accuracy', 
                                          'Weighted precison', 
                                          'Weighted recall', 
                                          'Weighted f1'], ha='center', rotation=0)
    plt.legend(ncol=2)                            
    return fig


def log_mean_conf_matrix(mean_conf_matrix, save_fig=False):
    for e in range(mean_conf_matrix.shape[0]):
        conf_fig = plot_confusion_matrix(None, None, CLASS_TO_GEST, title=f'Epoch {e}', matrix=mean_conf_matrix[e,:,:], return_fig=True,
                                         cmap='PuBuGn')
        if save_fig:
            conf_fig_file = f"ep_{wandb.config['n_epochs']}_average_conf_matrix_taum_{wandb.config['tau_mem']}_tausyn_{wandb.config['tau_syn']}.png"
            conf_fig.savefig(os.path.join(SNN_FIG, conf_fig_file), dpi=300, bbox_inches='tight')
        wandb.log({f"avg_conf_matrix": wandb.Image(conf_fig.get_figure())})




def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, class_to_gesture: dict,
                          title="", matrix=None, annotate=False, save_fig =False, 
                          filename_prefix="", filename_suffix="", return_fig = False, cmap='flare'):
    if matrix is None:
        matrix = confusion_matrix(actual_classes, predicted_classes, normalize="true")
    df_cm = pd.DataFrame(matrix, list(class_to_gesture.values()), list(class_to_gesture.values()))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns_plot = sns.heatmap(df_cm, cmap=cmap, annot=True, annot_kws={"size": 8}, fmt=".2f", ax=ax, 
                           vmin=0, vmax=1, cbar_kws={'label': 'Normalized Predictions Ratio','pad': 0.01})


    if annotate:
        for t in sns_plot.texts:
            if float(t.get_text()) > 0:
                # if the value is greater than 0.4 then I set the text
                t.set_text(t.get_text())
            else:
                t.set_text("")  # if not it sets an empty text

    sns_plot.set_xticklabels(sns_plot.get_xmajorticklabels(),   rotation=0)
    sns_plot.set_yticklabels(sns_plot.get_ymajorticklabels(), rotation=0)
    plt.xlabel("Predicted Gesture")
    plt.ylabel("True Gesture")

    # plt.show()
    if save_fig:
        conf_fig_file = f"{filename_prefix}_conf_matrix_linear_{filename_suffix}.png"
        fig.savefig(os.path.join(CLF_FIG, conf_fig_file), dpi=300, bbox_inches='tight')
        logging.info(f"Saved fig in {os.path.join(CLF_FIG, conf_fig_file)}")

    if return_fig:
        return fig