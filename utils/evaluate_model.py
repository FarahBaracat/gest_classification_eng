from constants import CLASS_TO_GEST
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from snntorch import functional as SF
import wandb
from typing import List



def compute_metric(y_true, y_pred, metric_name, class_names, avg='weighted'):
    if metric_name == 'acc':
        return balanced_accuracy_score(y_true, y_pred)

    elif metric_name == 'f1':
        return f1_score(y_true, y_pred, average=avg)

    elif metric_name == 'confusion_matrix':
        return confusion_matrix(y_true, y_pred)

    elif metric_name == 'clf_report':
        return classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    elif metric_name == 'precision_recall_fscore_support':
        return precision_recall_fscore_support(y_true, y_pred, average=avg)



def get_perf_results(clf, results_dict, x_train, y_train, x_test, y_test, 
                     datasets:List[str]=['train', 'val']):
    average = "weighted"  # 'macro'
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    for dataset in datasets:
        if dataset == "train":
            y_true = y_train
            y_pred = y_pred_train

        if dataset == "val":
            y_true = y_test
            y_pred = y_pred_test

        results_dict["acc_" +
                     dataset].append(balanced_accuracy_score(y_true, y_pred))
        results_dict["f1_" +
                     dataset].append(f1_score(y_true, y_pred, average=average))
        results_dict["conf_matrix_" +
                     dataset].append(confusion_matrix(y_true, y_pred, normalize="true"))
        results_dict["class_rep_" +
                     dataset].append(classification_report(y_true, y_pred))
        
        prec, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, average=average)
        results_dict["prec_" +dataset].append(prec)
        results_dict["recall_" +dataset].append(recall)

    return results_dict



def forward_test_data(data_loader, net, device, loss_fn, return_rec=False):
  """
  computes the accuracy for the test set
  """
  
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    loss_tracker = []

    rec_tracker = []
    y_pred, y_true = [], []
    # Pass through the test loader
    for data, targets in iter(data_loader):
      data = data.to(device)
      targets = targets.to(device)
      rec = net.forward(data)

      batch_size = rec['spk1'].size(1)
      acc += SF.accuracy_rate(rec['spk1'], targets) * batch_size
      total += batch_size

      # compute loss
      loss = loss_fn(rec['spk1'], targets)
      loss_tracker.append(loss.item())

      # get predicted label
      _, pred = rec['spk1'].sum(dim=0).max(1)
      y_pred.extend(pred.cpu().numpy())
      y_true.extend(targets.cpu().numpy())

      

      rec_tracker.append(rec)

    # pool the results
    e_acc = acc/total
    print(f"Accuracy: {acc}   total:{total}")
    # print(f"y_pred:{y_pred}   \ny_true:{y_true}")

    e_loss = np.sum(loss_tracker)/(len(loss_tracker) *batch_size)

    if return_rec:

        return e_acc, e_loss, rec_tracker, y_pred, y_true
    else:
        return e_acc, e_loss, y_pred, y_true
    



def organize_mean_sd_gesture_df(mean_conf_matrix:np.ndarray, std_conf_matrix:np.ndarray):
    """
    Organize the mean and std of the confusion matrix for each gesture in a dataframe
    """
    mean_sd_gesture_df = pd.DataFrame()
    mean_sd_gesture_df['mean_corr_pred'] = np.diag(mean_conf_matrix)
    mean_sd_gesture_df['std_corr_pred'] = np.diag(std_conf_matrix)
    mean_sd_gesture_df['class_labels']= list(CLASS_TO_GEST.values())
    return mean_sd_gesture_df


def compute_mean_std_for_array(metric_matrix, axis):

    mean_metric = metric_matrix.mean(axis=axis)
    std_metric = metric_matrix.std(axis=axis)
    return mean_metric, std_metric



