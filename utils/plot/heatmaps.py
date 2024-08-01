from constants import *
import utils.preprocess_analog as pre

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging


def plot_heatmap(df, title, ax, cbar_label=None, vmin=None, vmax=None, cbar=False, ylabel='Channels', xlabel='Reps',
                 yticks_label=None,  xticks_label=None, yticks_step=5, xticks_step=5, hor_lines=None, cmap='rocket_r'):
    # cmap = sns.color_palette("#69d", as_cmap=True)
    hmap = sns.heatmap(df, ax=ax, cbar_kws={ 'label': cbar_label}, 
                       cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar)  # 'shrink': 1,'aspect': 50
    fontsize = 10

    if cbar:
        cbar = hmap.collections[0].colorbar
        cbar.set_label(cbar_label, labelpad=5, fontsize=fontsize)

    ax.set_title(title, fontsize=fontsize, pad=0.3)

    if hor_lines is not None:
        ax.hlines(hor_lines, *ax.get_xlim(),
                  color=COLOR_DICT['naval'], linewidth=0.3, linestyle="dotted")

    if yticks_label is not None:
        ax.set_yticks(np.arange(0.5, len(df.index)+0.5, yticks_step), minor=False)
        hmap.set_yticklabels(
            yticks_label[::yticks_step], rotation=0, fontsize=fontsize)
    if xticks_label is not None:
        ax.set_xticks(np.arange(0.5, len(df.columns)+0.5, xticks_step), minor=False)
        hmap.set_xticklabels(xticks_label[::xticks_step], fontsize=fontsize, rotation=0)

    else:
        hmap.set_xticklabels(rotation=0)

    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    # plt.show()




def create_hmap_across_tasks(eng_dataset, chs_to_exclude=None, stat='rms', vmin=None, vmax=None):

    df = eng_dataset.filt_df
    task_labels = eng_dataset.task_order


    # plot the heatmap
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(nrows=5, ncols=3)

    if chs_to_exclude is not None:
        df = pre.select_channels(df, chs_to_exclude)

    cbar_label = None
    if stat == 'rms' or stat == 'mav':
        cbar_label = f'{stat} (uV)'
    elif stat == 'pow':
        cbar_label = f'{stat} (uV^2)'

    for task_id in range(N_TASKS):
        flex_power_df, ext_power_df, rest_power_df = pre.extract_feat_for_task(eng_dataset, task_id, feat=stat)
        title_flx = None
        title_ext = None
        title_rest = None

        if task_id == 0:
            title_flx = 'Close'
            title_ext = 'Open'
            title_rest = 'Rest'

        row = task_id  # int(np.floor(task_id/2))
        col = 0  # 2* (task_id%2)

        if vmin is None:
            min_scale = min(flex_power_df.min().min(),
                            ext_power_df.min().min())
            max_scale = max(flex_power_df.max().max(),
                            ext_power_df.max().max())
            is_scale_str = 'unscaled'
        else:
            min_scale = vmin
            max_scale = vmax
            is_scale_str = 'scale'

        ax1 = fig.add_subplot(gs[row, col])
        plot_heatmap(flex_power_df, title_flx, ax1, cbar_label=cbar_label, cbar=True,
                     ylabel=f'task {task_id}:{task_labels[task_id]}', vmin=vmin, vmax=vmax)

        ax2 = fig.add_subplot(gs[row, col + 1])
        plot_heatmap(ext_power_df, title_ext, ax2, cbar_label=cbar_label,
                     vmin=min_scale, vmax=max_scale, cbar=True, ylabel=None)

        ax3 = fig.add_subplot(gs[row, col + 2])
        plot_heatmap(rest_power_df, title_rest, ax3, cbar_label=cbar_label,
                     vmin=min_scale, vmax=max_scale, cbar=True, ylabel=None)

        for ax in [ax1, ax2, ax3]:
            if task_id < N_TASKS-1:
               ax.tick_params(axis='x', which='both', length=0)
               ax.set_xlabel('')
               ax.set_xticklabels([])

    # add title
    # fig.text(0.5, 0.91, f"Chs {stat} [{pipeline['bp_cutoff_freq'][0]} - {pipeline['bp_cutoff_freq'][1]}]",  ha='center', va='center', fontsize=7)
    plt.show()
    fig.text(0.5, 0.91, f" ",  ha='center', va='center', fontsize=7)

    if eng_dataset.save_figs:
        filename = f"day{eng_dataset.day}{eng_dataset.session}_{stat}_wrest_exclude_bad_ch_{is_scale_str}_heatmap_filt_{eng_dataset.filt_pipeline['bp_cutoff_freq'][0]}_{eng_dataset.filt_pipeline['bp_cutoff_freq'][1]}.png"
        # save the figure
        logging.info(f"Saving figure to: {filename}")
        fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')

