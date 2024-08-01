import wandb
import os
import sys
script_dir = os.getcwd()

# if os.path.join(script_dir, 'revisit_code_2023') not in sys.path:
#     sys.path.insert(0, os.path.join(script_dir, 'revisit_code_2023'))

from utils.svm import create_feat_df, train_svm, train_in_time, create_figures_dir
from utils.plotting import plot_mean_acc_per_class
from constants import DAY, SESSION, POST_PROC_DIR, EXT_PHASE, FLX_PHASE

# import os
import pickle as pkl
from collections import Counter, namedtuple


params_defaults = dict(is_temporal=True,
                       seed= 1, 
                       k_cv= 5,
                       kernel ='rbf',
                       bin_width=0.1,  # duration in sec
                       overlap_perc = 0.5, # percentage of overlap
                       use_mov_avg = True,   # used to define the saved files prefix
                       C=1,
                       gamma=0.01,
                       clf_classes = 'flx_vs_ext_separate'

                    )

wandb.init(project="eng_2023_svm_hyper",entity="farahbaracat", config=params_defaults)

# Training parameters
SELECT_CLASS  = [1,2,6,8]  # in case of non-temporal


SAVE_FIG = True

Gest = namedtuple('gesture', ['id', 'phase'])
sel_gest = [Gest(0,EXT_PHASE),Gest(1,FLX_PHASE), Gest(2,FLX_PHASE), Gest(3,FLX_PHASE),Gest(4,FLX_PHASE)] 


def train():
    print("Training SVM")

    # create directory for figures
    create_figures_dir()


    file_post = os.path.join(POST_PROC_DIR, f'day{DAY}{SESSION}_data_post.pkl')

    with open(file_post, 'rb') as f:
        data_post = pkl.load(f)


    # unpacking data
    trig_df_post = data_post['trig_df_post']
    filt_df = data_post['filt_df']
    reps_count_dict = data_post['n_reps']

  

    if wandb.config['is_temporal']:
        bin_stat = 'pow'
        selected_class = [0,1,3,4]
        results_df, mean_conf, std_conf = train_in_time(filt_df, trig_df_post, reps_count_dict, sel_gest, wandb.config['bin_width'], 
                                                        bin_stat, selected_class, plot_feat=False, seed=wandb.config['seed'],
                                                        k_cv=wandb.config['k_cv'], save_fig=SAVE_FIG, kernel=wandb.config['kernel'],
                                                        C = wandb.config['C'], gamma=wandb.config['gamma'])
                

    else:
        # when training on all the trial, it is better to use k_cv=3 because the test set is too small 
        feat_df, labels = create_feat_df(filt_df, trig_df_post, reps_count_dict=reps_count_dict, feat_list=['pow'], clf_classes=wandb.config['clf_classes'])
        print(f"\n{Counter(feat_df['label'])}\n{labels}")
        results_df = train_svm(feat_df, labels, SELECT_CLASS, k_cv=wandb.config['k_cv'], save_fig=SAVE_FIG, annotate_cm=False,
                               seed=wandb.config['seed'], return_conf_test=False,
                               kernel=wandb.config['kernel'], C=wandb.config['C'], gamma=wandb.config['gamma'])
    print("\nResults\n------------------")
    print(results_df)


    

if __name__ == "__main__":
    train()