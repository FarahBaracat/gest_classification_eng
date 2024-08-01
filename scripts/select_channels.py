from constants import DAY, SESSION, POST_PROC_DIR, TIME_VAR
from utils.preprocess_analog import remove_bad_chs
from utils.pca import scale_dataset, compute_pearson_corr, sort_correlation_matrix, filter_significant_corr, select_top_corr_channels
import os
import pickle as pkl
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

logging.getLogger().setLevel(logging.INFO)


SIGNIFICANCE_LVL= 0.05   # significance level for correlation between PC and channels
N_TOP_CH = 10
N_TOP_PC = 5


def select_channels():
    file_post = os.path.join(POST_PROC_DIR, f'day{DAY}{SESSION}_data_post.pkl')

    with open(file_post, 'rb') as f:
        data_post = pkl.load(f)


    # unpacking data
    trig_df_post = data_post['trig_df_post']
    filt_df = data_post['filt_df']
    n_reps = data_post['n_reps']

    logging.info(f"Loaded filtered data:{filt_df.shape}\ncolumns:{filt_df.columns}")

    # select channels
    filt_df = filt_df.drop(TIME_VAR, axis=1)
    filt_df = remove_bad_chs(filt_df)
    logging.debug(f"Filtered data after removing bad channels:{filt_df.shape}\ncolumns:{filt_df.columns}")

    # scale data prior to PCA: subtract mean and divide by std
    filt_df_scaled = scale_dataset(filt_df)
    logging.info(f"Scaled df:{filt_df_scaled.mean(axis=0)} \n{filt_df_scaled.std(axis=0)}")

    n_valid_channels = filt_df_scaled.shape[1]
    n_pc = n_valid_channels
    pca = PCA(n_components=n_pc)
    data_pca = pca.fit_transform(filt_df_scaled)

    corr_matrix, p_values = compute_pearson_corr(filt_df_scaled, data_pca, n_valid_channels, n_pc, SIGNIFICANCE_LVL)
    significant_corr_matrix = filter_significant_corr(corr_matrix, p_values, SIGNIFICANCE_LVL)
    sorted_corr, sorted_idx_des = sort_correlation_matrix(significant_corr_matrix)


    selected_ch = select_top_corr_channels(N_TOP_CH, N_TOP_PC, sorted_idx_des)
    unique_channels = np.unique(selected_ch)
    cum_expvar = np.sum(pca.explained_variance_ratio_[:N_TOP_PC]) * 100
    print("Unique channels:{} \n{}".format(len(unique_channels), unique_channels))
    print("\nCum explained var:{}".format(cum_expvar))


    # plot sorted correlation matrix
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    sns.heatmap(sorted_corr, cmap="flare", ax=ax,vmin=0, vmax=1)
    plt.title("Sorted correlation matrix")
    
    plt.tight_layout()
    ax = fig.add_subplot(122)
    sns.heatmap(significant_corr_matrix[::-1], cmap="flare", ax=ax, 
                vmin=0, vmax=1)
    plt.title("Significant correlation matrix")

    plt.show()


if __name__ == '__main__':
    select_channels()