
from constants import *

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def plot_ch_annotate_trials(eng_dataset, ch, y_range, title, data_type='filt'):
    if data_type == 'filt':
        df = eng_dataset.filt_df
    elif data_type == 'post':
        df = eng_dataset.post_data_df
    else:
        raise ValueError("Data type should be either 'filt' or 'post'")

    trig_df = eng_dataset.trig_df
    task_order = eng_dataset.task_order
    n_tasks  = eng_dataset.n_tasks

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    plot = ax.plot(df[TIME_VAR], df[ch], label= f"Ch {ch}")
    plt.box(False)

    for rep_id, trig in enumerate(trig_df[TRIG_VAR]):
        if rep_id == len(trig_df[TRIG_VAR]) - 1:
            box_width = df[TIME_VAR].iloc[-1] - trig_df[TRIG_VAR][rep_id]
        else:
            box_width = trig_df[TRIG_VAR][rep_id+1] - trig_df[TRIG_VAR][rep_id]
        facecolor = list(COLOR_DICT.values())[int(trig_df.iloc[rep_id]['task_id'])]
        box = Rectangle((trig, y_range[0]),box_width, np.abs(y_range[0])+y_range[1], facecolor=facecolor, alpha=0.4,
                        ) #label=TASK_ORDER[int(trig_df.iloc[rep_id]['task_id'])]
        ax.axvline(x=trig,ymin= df[ch].min(), ymax= df[ch].max(), color='r', linestyle='--', alpha=0.2)
        ax.add_patch(box)

    # add legend
    legend_patches = []
    for task in range(n_tasks):
        legend_patches.append(mpatches.Patch(color=list(COLOR_DICT.values())[task], label=task_order[task], alpha=0.4))
    plt.legend(handles=legend_patches, ncol=5)
    
    plt.tight_layout()
    # plt.legend(handles=box.legend_elements()[0], labels=TASK_ORDER, ncol=5)

    plt.xlabel('Time [sec]', fontsize=8)
    plt.ylabel(r'Amplitude [$\mu$V]', fontsize=8)
    plt.title(title)
    plt.xlim([0,245])
    plt.show()

    if eng_dataset.save_figs:
        base_filename = f"day{eng_dataset.day}{eng_dataset.session}_{data_type}_ch_{ch}"
        if data_type=='filt':
            fill = f"{eng_dataset.filt_pipeline['bp_cutoff_freq'][0]}_{eng_dataset.filt_pipeline['bp_cutoff_freq'][1]}"
        if data_type=='post':
            fill = 'raw'
        fig.savefig(os.path.join(FIG_DIR, f"{base_filename}_{fill}_eng_cut_annotate_reps.png"),
                     dpi=300, bbox_inches='tight')


