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
    snn_color = sns.color_palette(palette='Accent')[0]

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.hist(w_learnt.flatten(), bins=100, alpha=trained_alpha, label='Trained', 
            color=snn_color)
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
        filename = f"{day}_ep_{n_epochs}_trained_vs_init_wdist"
        file_path = os.path.join(SNN_FIG, f"{filename}.png") 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")