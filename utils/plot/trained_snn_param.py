import wandb
import os

from constants import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging



def plot_learnt_wdist(day:int, w_learnt:np.ndarray, w_init:np.ndarray, save_fig=False):
    
    n_epochs = wandb.config['n_epochs']
    # Figure styling
    trained_alpha = 0.6
    init_alpha = 0.4
    # snn_color = sns.color_palette(palette='Accent')[0]

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.hist(w_learnt.flatten(), bins=100, alpha=trained_alpha, label='Trained', 
            color=COLOR_DICT['dark_cyan'])
    if w_init is not None:
        ax.hist(w_init.flatten(), bins=100, alpha=init_alpha, label='Initial', 
                color=COLOR_DICT['midnight_blue'])

    ax.set_xlabel('Weight value (a.u.)', labelpad=XLAB_PAD)
    ax.set_ylabel('Frequency', labelpad=YLAB_PAD)
    sns.despine(ax=ax, offset=0, trim=False)
    # fig.legend(loc='upper left', ncol=1, frameon=False)
    plt.legend(frameon=False)
    fig.tight_layout()


    if save_fig:
        filename = f"{day}_ep_{n_epochs}_trained_vs_init_wdist_taum_{wandb.config['tau_mem']}_tausyn_{wandb.config['tau_syn']}"
        file_path = os.path.join(SNN_FIG, f"{filename}.png") 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")



def annotate_tip_bar(ax, y_value:np.ndarray, values_to_annotate:np.ndarray, color:str, padding:float=None):
    """
    Adds the value of tau to the tip of the bar
    """
    x_locs = [rect.get_x() for rect in ax.patches]
    for i, tau in enumerate(values_to_annotate):
        if padding:
            y_value[i] += padding
        plt.text(x_locs[i],y_value[i], f"{tau:.2f}", fontsize=8, 
                 color=color)
        


def plot_learnt_threshold(day:int, vth_learnt:np.ndarray, vth_init:np.ndarray, save_fig=False):

    n_neurons = len(vth_learnt) 
    n_epochs = wandb.config['n_epochs']

    # Figure styling
    # set y limit based on the parameter max value
    width = 0.45
    # snn_color = '#028189' #sns.color_palette(palette='Accent') [0]
    ylim = np.max(vth_learnt) * 1.1
    alpha_trained = 0.5
    alpha_init = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        

    ax.bar(np.arange(n_neurons), vth_learnt, width=width, alpha=alpha_trained, 
           color= COLOR_DICT['dark_cyan'], label='Trained')
    
    ax.bar(np.arange(n_neurons), vth_init, width=width, alpha=alpha_init, 
            color= COLOR_DICT['midnight_blue'], label='Initial')
    ax.set_xlim(-0.5, n_neurons-0.5)
    ax.set_xlabel("Output Neuron", labelpad=XLAB_PAD)
    ax.set_xticks(np.arange(0, n_neurons, 1), list(CLASS_TO_GEST.values()))

    ax.set_ylim(0, ylim)
    sns.despine(ax=ax, offset=0, trim=False)

    plt.legend(frameon=False)
  

    # select the maximum value of learnt_tau and init_tau to use for annotation
    annot_tip = np.maximum.reduce([vth_learnt, vth_init])
    annotate_tip_bar(ax, annot_tip, vth_learnt, COLOR_DICT['midnight_blue'], padding=0.01)

    ax.set_ylabel(r"$U_{thr}$ (V)", labelpad=YLAB_PAD)

    if save_fig:
        filename = f"{day}_ep_{n_epochs}_trained_vs_init_threshold_taum_{wandb.config['tau_mem']}_tausyn_{wandb.config['tau_syn']}"
        file_path = os.path.join(SNN_FIG, f"{filename}.png") 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")

