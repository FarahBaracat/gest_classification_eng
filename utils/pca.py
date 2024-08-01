from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def compute_pearson_corr(data_df_scaled, data_pca, n_ch, n_pc):
    corr_matrix = np.empty((n_ch, n_pc))
    p_values = np.empty_like(corr_matrix)
    for comp in range(n_pc):  # loop over PC
        for ch in range(n_ch): 
            corr_matrix[ch, comp], p_values[ch, comp] = pearsonr(data_pca[:, comp], data_df_scaled[:, ch])
    return corr_matrix, p_values


def sort_correlation_matrix(significant_corr_matrix):
    sorted_idx = pd.DataFrame(np.argsort(significant_corr_matrix, axis=0))
    sorted_idx_des = sorted_idx.reindex(index=sorted_idx.index[::-1])
    sorted_corr = pd.DataFrame(np.sort(significant_corr_matrix, axis=0))
    
    # sort in descending order
    sorted_corr_des = sorted_corr.reindex(index=sorted_corr.index[::-1])
    return sorted_corr_des, sorted_idx_des


def filter_significant_corr(corr_matrix, p_values, stat_sig_lvl):
    is_significant = p_values < stat_sig_lvl
    significant_corr_matrix = np.abs(corr_matrix * is_significant)
    return significant_corr_matrix


def select_top_corr_channels(top_n_chs, top_n_pcs, sorted_corr_ids):
    sliced_corr = sorted_corr_ids.iloc[:top_n_chs, :top_n_pcs]
    return sliced_corr

