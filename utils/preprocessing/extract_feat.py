
from utils.preprocess_analog import get_single_rep_for_task, split_rep_to_flex_ext, extract_rest_for_rep
from srcs.engdataset import ENGDataset
from constants import *

import logging
import numpy as np
import pandas as pd


def compute_rms(df: pd.DataFrame):
    """
    Computes the root mean square of the data
    """
    return pd.DataFrame(np.sqrt(np.mean(np.square(df), axis=0)))


def compute_power(df: pd.DataFrame):
    """
    Computes the power of the data
    """
    return pd.DataFrame(np.mean(np.square(df), axis=0))


def compute_mav(df: pd.DataFrame):
    """
    Computes the mean absolute value of the data
    """
    return pd.DataFrame(np.mean(np.abs(df), axis=0))

def compute_full_wave(df: pd.DataFrame):
    """
    Computes the full wave rectification of the data
    """
    return pd.DataFrame(np.mean(np.abs(df), axis=0))

def compute_overlap_wstime(wind_size: float, overlap_perc: float, duration: float):
    """
    Computes the start time in seconds of each window in a sliding window approach."""
    stride = wind_size*(1-overlap_perc)
    start_ar = np.arange(0, duration - wind_size, stride)

    if np.round(start_ar[-1] + stride + wind_size,4) <= duration:
        start_ar = np.append(start_ar, start_ar[-1] + stride)
    return start_ar


def extract_feature_for_task_rep(eng_dataset:ENGDataset, task_id:int, rep_id:int, feature:str, 
                                 wind_size:float, overlap_perc:float, return_win_stime:bool=False):
    #TODO: add a case where we are not extracting over windows but over the whole rep
    """
    Extracts features from the given task and rep. Splits the rep into flexion, extension and rest phases first
    Returns the feature dataframe for each phase of shape (n_windows, n_channels + [Time, feat_win, rep_id] )
    """
    rep_df,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep_id, task_id)
    assert rep_df.shape[0] > 0, f"Empty repetition {rep_id} for task {task_id}"
    rep_df_flx, rep_df_ext = split_rep_to_flex_ext(rep_df, rep_st, rep_et, eng_dataset)
    _,_ ,rep_df_rest = extract_rest_for_rep(eng_dataset, task_id, rep_id)
    try:
        logging.debug(f"task_id: {task_id}  rep {rep_id}:  flx:{rep_df_flx[TIME_VAR].iloc[-1] - rep_df_flx[TIME_VAR].iloc[0]}\next:{rep_df_ext[TIME_VAR].iloc[-1] - rep_df_ext[TIME_VAR].iloc[0]}")
    except:
        print("Error in extracting rep duration")
   
    flx_feat_df = extract_feature_for_rep(rep_df_flx, rep_id, feature, wind_size, overlap_perc, eng_dataset.fs, return_win_stime)
    ext_feat_df = extract_feature_for_rep(rep_df_ext, rep_id, feature, wind_size, overlap_perc, eng_dataset.fs, return_win_stime)
    rest_feat_df = extract_feature_for_rep(rep_df_rest, rep_id, feature, wind_size, overlap_perc, eng_dataset.fs, return_win_stime)

    return flx_feat_df, ext_feat_df, rest_feat_df


def extract_feature_for_rep(rep_df: pd.DataFrame, rep_id:int, feature: str, wind_size: float, overlap_perc:float, 
                            eng_famp:int, return_win_stime:bool=False, clip_feature:bool=False):
    """
    Extracts features from the data using a sliding window approach.
    win_size: window size in seconds
    """
    # Each function takes df with n_ch columns, n_samples rows
    feat_funct= {'rms': compute_rms, 'mav': compute_mav,
                 'power': compute_power,'full_wave': compute_full_wave}  

  
    # check if there is a time column, if yes strip it
    if TIME_VAR in rep_df.columns:
        time_arr = rep_df[TIME_VAR].to_numpy()
        rep_df = rep_df.drop(TIME_VAR, axis=1)
        # TODO: assert later that time_arr is the same as the added TIME

    n_samples, n_ch = rep_df.shape[0], rep_df.shape[1]
    logging.debug(f"Single trial: {n_samples} samples, {n_ch} channels")
    duration = n_samples/eng_famp # in seconds

    # compute the start time of each window
    start_ar = compute_overlap_wstime(wind_size, overlap_perc, duration)
    logging.debug(f"Extracting features from {len(start_ar)} windows")
    feat_df_ar = pd.DataFrame() # columns: time, all channels, window_number
    for win, win_stime in enumerate(start_ar):
        # compute the start and end index of the window
        win_sidx = int(win_stime* eng_famp)
        win_eidx = int(win_sidx + wind_size* eng_famp)
        # extract the window
        win_df = rep_df.iloc[win_sidx:win_eidx, :]
        # compute the features
        feat_df = feat_funct[feature](win_df).T
        # add the time and window number
        feat_df[TIME_VAR] = win_stime
        feat_df[FEAT_WIN_COL] = win
        feat_df[REP_ID_COL] = rep_id
        feat_df_ar = pd.concat([feat_df_ar, feat_df], axis=0, ignore_index=True)

    # TODO: clip feature
    # if clip_feature:
    #     feat_df_ar = clip_feature_to_mean(feat_df_ar, config)
    if return_win_stime:
        return feat_df_ar, start_ar
    else:
        return feat_df_ar





# def get_stat_moving_wind_for_rep_id(eng_dataset, rep_id, task_id, wind_duration, stride_in_sec, min_periods=0, feat='pow', combine_flx_ext=False):
#     """
#     Compute a statistic over a moving window of size wind_size
#     """
#     #CHECKTHIS: check on the implementation fo moving average    
#     # get start and end of the rep
#     rep_df,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep_id, task_id)
#     rep_df_flx, rep_df_ext = split_rep_to_flex_ext(rep_df, rep_st, rep_et, eng_dataset)
#     _,_ ,rep_df_rest = extract_rest_for_rep(eng_dataset, task_id, rep_id)

#     print(f"task_id: {task_id}  rep {rep_id}:  flx:{rep_df_flx[TIME_VAR].iloc[-1] - rep_df_flx[TIME_VAR].iloc[0]}\next:{rep_df_ext[TIME_VAR].iloc[-1] - rep_df_ext[TIME_VAR].iloc[0]}")
#     # convert stride in sec to stride in samples
#     stride = int(stride_in_sec * ENG_FS)
#     wind_size = int(wind_duration * ENG_FS)
#     min_periods = int(min_periods * ENG_FS)

#     if combine_flx_ext:
#         # flx_ext_df columns 56 + 1 time var
#         flx_ext_df = pd.concat([rep_df_flx, rep_df_ext], axis=0, ignore_index=True, sort=False)

#         print(f"stride: {stride}  wind_size: {wind_size} n_windows: {(flx_ext_df.shape[0] - wind_size) // stride +1}")
#         n_wind = (flx_ext_df.shape[0] - wind_size) // stride +1
#         flx_ext_wind_avg_df = get_avg_moving_window(flx_ext_df, wind_size, stride, min_periods, feat)
#         assert flx_ext_wind_avg_df.shape[0] == n_wind, "check n_wind"
#         return flx_ext_wind_avg_df


#     else:
#         # flx_df columns 56 + 1 time var
#         flx_wind_avg_df = get_avg_moving_window(rep_df_flx, wind_size, stride, min_periods, feat)
#         ext_wind_avg_df = get_avg_moving_window(rep_df_ext, wind_size, stride, min_periods, feat)
#         rest_wind_avg_df = get_avg_moving_window(rep_df_rest, wind_size, stride, min_periods, feat)

#         # check on n_wind
#         for df, df_avg in zip([rep_df_flx, rep_df_ext, rep_df_rest],[flx_wind_avg_df,ext_wind_avg_df,rest_wind_avg_df]):
#             n_wind = (df.shape[0] - wind_size) // stride + 1
#             # assert df_avg.shape[0] == n_wind, f"check n_wind: {n_wind} vs. {df_avg.shape[0] }"
  
#         return flx_wind_avg_df, ext_wind_avg_df, rest_wind_avg_df
