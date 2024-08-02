from constants import *
from srcs.engdataset import ENGDataset
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy import stats
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from typing import List

def remove_bad_chs(df:DataFrame, bad_ch_list:List):
    return df.drop(bad_ch_list, axis=1)


def get_abs_ch_id(ch_id_after_bad_chs, list_of_bad_chs):

    "After removal of bad channels, the channel ids are not consecutive. This function returns the absolute channel ids"
    
    abs_ch_list= np.array(ch_id_after_bad_chs)
    for ch in list_of_bad_chs:
        abs_ch_list[abs_ch_list >= ch] += 1

    return  abs_ch_list


# select all channels except listed
def select_channels(df, chs_to_exclude):
    return df[[col for col in df.columns if col not in chs_to_exclude]]



def scale_dataset(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled



# def get_single_rep_for_task(eng_dataset:ENGDataset, rep_id:int, task_id:int):
#     """
#     Each rep is 5 sec, starts with flexion for 1 sec, followed by extension 1 sec and then 3 sec of rest
#     """
#     # check is_last correct
#     # assert rep_id < n_reps[task_id], f"rep_id {rep_id} is larger than n_reps {n_reps[task_id]} for task {task_id}"
#     trig_df = eng_dataset.trig_df
#     df = eng_dataset.filt_df
#     task_rep_count = eng_dataset.task_rep_count
#     rep_start, rep_end = get_trig_start_end_for_task(df, trig_df, rep_id, task_id, task_rep_count)

        
#     # slice the rep
#     rep_df = df[(df[TIME_VAR] >= rep_start) & (df[TIME_VAR] < rep_end)]
#     logging.debug(f"rep {rep_id} start: {rep_start}, end: {rep_end}  shape:{rep_df.shape}")

#     return rep_df, rep_start, rep_end



# def get_trig_start_end_for_task(df:DataFrame, trig_df:DataFrame, rep_id:int, task_id:int, n_reps_dict:dict[int,int]):
#     """
#     Gets the start and end triggers for a given task_id and rep_id
#     """
#     is_last = False

#     if rep_id == n_reps_dict[task_id] -1 :
#         is_last = True

#     # get the start and end time of the rep
#     rep_start = trig_df[trig_df['task_id'] == task_id][TRIG_VAR].iloc[rep_id]
    
#     # Last repetition needs special handling
#     if is_last:
#         if task_id == N_TASKS-1: # if last rep and task then crop till end of recording
#             rep_end = df[TIME_VAR].iloc[-1]
#         else: # if last rep only, then the end matches the start of the following task
#             rep_end = trig_df[trig_df['task_id'] == task_id+1][TRIG_VAR].iloc[0]
#     else:
#         rep_end = trig_df[trig_df['task_id'] == task_id][TRIG_VAR].iloc[rep_id + 1]
#     return rep_start,rep_end


def get_start_end_time(rep_df):
    return rep_df[TIME_VAR].iloc[0], rep_df[TIME_VAR].iloc[-1]


def get_pow(arr_1d):
    return np.mean((arr_1d**2))


def get_mav(arr_1d):
    return np.mean(np.abs(arr_1d))


def get_bin_stat(time_col, data_col, bin_range, bin_stat, bin_width):
    if bin_stat == 'pow':
        bin_stat, bin_edges, binnumber = stats.binned_statistic(time_col, values=data_col ,bins=bin_range, statistic=get_pow)
    elif bin_stat == 'mav':
        bin_stat, bin_edges, binnumber = stats.binned_statistic(time_col, values=data_col ,bins=bin_range, statistic=get_mav)

    return bin_stat, bin_edges, binnumber


# def extract_rest_for_rep(eng_dataset, task_id, rep_id):
#     """
#     Extract the rest period for a given rep. This is the middle 1 sec before the start of the repetiton (flexion part).
#     Each repetition is preceded by 3 secs of rest. Here, I extract the middle 1 sec of rest.
#     """
#     n_reps_dict =eng_dataset.task_rep_count
#     df = eng_dataset.filt_df
#     if rep_id ==0:
#         if task_id ==0:
#             _,rep_st, rep_et = get_single_rep_for_task(eng_dataset,rep_id, task_id)
#             rest_start = rep_st - 2 # remove 2 seconds from start of first rep
#             rest_end = rest_start + 1 # get only 1 sec of rest

#         else:
#             _,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep_id=n_reps_dict[task_id-1]-1, task_id=task_id-1)
#             rest_start = rep_et - 2 # remove 2 seconds from start of first rep
#             rest_end = rest_start + 1 # get only 1 sec of rest
#     else:
#         _,rep_st, rep_et = get_single_rep_for_task(eng_dataset,rep_id, task_id)
#         rest_start = rep_st - 2 # remove 2 seconds from start of first rep
#         rest_end = rest_start + 1 # get only 1 sec of rest
    
#     # slice df
#     rest_df = df[(df[TIME_VAR] >= rest_start) & (df[TIME_VAR] < rest_end)]
    
#     return rest_start, rest_end,rest_df


# def split_rep_to_flex_ext(rep_df:DataFrame, rep_st:float, rep_et:float, eng_dataset:ENGDataset):
#     """
#     Splits a single rep into its flexion and extension segments. 
#     Extension segment is right after the flexion segment. Each rep ends with 3 sec of rest, the ext seg is right preceding that.
#     """
#     # note that condition  (rep_df[TIME_VAR] >= rep_st) is for completeness. The rep_df always starts at rep_st
#     rep_df_flex = rep_df[(rep_df[TIME_VAR] >= rep_st) &(rep_df[TIME_VAR] < rep_st + eng_dataset.flex_dur)]
#     # rep_df_ext = rep_df[(rep_df[TIME_VAR] >= rep_st + eng_dataset.flex_dur) & (rep_df[TIME_VAR] < rep_et - eng_dataset.rest_dur)]
#     rep_df_ext = rep_df[(rep_df[TIME_VAR] >= rep_st + eng_dataset.flex_dur) & (rep_df[TIME_VAR] < rep_st + eng_dataset.flex_dur + eng_dataset.ext_dur)]

#     return rep_df_flex, rep_df_ext


def get_avg_moving_window(data_df, wind_size, stride, min_periods, feat):
    """
    Compute the average of a moving window of size wind_size
    data_df includes channels and TIME_VAR columns
    """
    # extract the time dimension
    time_arr = data_df[TIME_VAR].to_numpy()

    #drop time column
    feat_df = data_df.drop(TIME_VAR, axis=1)
    if feat == 'pow':  # Todo: check on the columns, CHECKTHIS: why power I do it inside function?
        feat_df = feat_df.pow(2)

    stat_arr = feat_df.to_numpy()
    if wind_size >= stat_arr.shape[0]:
        logging.info("Chosen window size is larger or equal than the data size. Returning the mean of the entire data")
        window_avg = [np.mean(stat_arr,axis=0)]
        time_axis = [time_arr[0]]
 
    else:
        window_avg = [np.mean(stat_arr[i:i+wind_size,:],axis=0) for i in range(0, stat_arr.shape[0], stride)
                   if i+wind_size <= stat_arr.shape[0]+ min_periods]
        time_axis = [time_arr[i] for i in range(0, stat_arr.shape[0], stride)  if i+wind_size <= stat_arr.shape[0]+ min_periods]

    # convert back to df
    window_avg_df = pd.DataFrame(window_avg)
    window_avg_df[TIME_VAR] = time_axis
    return window_avg_df


def get_stat_moving_wind_for_rep_id(eng_dataset, rep_id, task_id, wind_duration, stride_in_sec, min_periods=0, feat='pow', combine_flx_ext=False):
    """
    Compute a statistic over a moving window of size wind_size
    """
    #CHECKTHIS: check on the implementation fo moving average
    df = eng_dataset.filt_df
    
    # get start and end of the rep
    rep_df,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep_id, task_id)
    rep_df_flx, rep_df_ext = split_rep_to_flex_ext(rep_df, rep_st, rep_et, eng_dataset)
    _,_ ,rep_df_rest = extract_rest_for_rep(eng_dataset, task_id, rep_id)

    print(f"task_id: {task_id}  rep {rep_id}:  flx:{rep_df_flx[TIME_VAR].iloc[-1] - rep_df_flx[TIME_VAR].iloc[0]}\next:{rep_df_ext[TIME_VAR].iloc[-1] - rep_df_ext[TIME_VAR].iloc[0]}")
    # convert stride in sec to stride in samples
    stride = int(stride_in_sec * ENG_FS)
    wind_size = int(wind_duration * ENG_FS)
    min_periods = int(min_periods * ENG_FS)

    if combine_flx_ext:
        # flx_ext_df columns 56 + 1 time var
        flx_ext_df = pd.concat([rep_df_flx, rep_df_ext], axis=0, ignore_index=True, sort=False)

        print(f"stride: {stride}  wind_size: {wind_size} n_windows: {(flx_ext_df.shape[0] - wind_size) // stride +1}")
        n_wind = (flx_ext_df.shape[0] - wind_size) // stride +1
        flx_ext_wind_avg_df = get_avg_moving_window(flx_ext_df, wind_size, stride, min_periods, feat)
        assert flx_ext_wind_avg_df.shape[0] == n_wind, "check n_wind"
        return flx_ext_wind_avg_df


    else:
        # flx_df columns 56 + 1 time var
        flx_wind_avg_df = get_avg_moving_window(rep_df_flx, wind_size, stride, min_periods, feat)
        ext_wind_avg_df = get_avg_moving_window(rep_df_ext, wind_size, stride, min_periods, feat)
        rest_wind_avg_df = get_avg_moving_window(rep_df_rest, wind_size, stride, min_periods, feat)

        # check on n_wind
        for df, df_avg in zip([rep_df_flx, rep_df_ext, rep_df_rest],[flx_wind_avg_df,ext_wind_avg_df,rest_wind_avg_df]):
            n_wind = (df.shape[0] - wind_size) // stride + 1
            # assert df_avg.shape[0] == n_wind, f"check n_wind: {n_wind} vs. {df_avg.shape[0] }"
  
        return flx_wind_avg_df, ext_wind_avg_df, rest_wind_avg_df


# TODO: remove not used
# def compute_stat_in_time_for_rep_id(df, trig_df, n_reps_dict, rep_id, task_id, flx_dur, ext_dur, rest_dur, bin_width, bin_stat, combine_flx_ext=False):
    
#     rep_df,rep_st, rep_et = get_single_rep_for_task(df, trig_df, rep_id, task_id, n_reps_dict)

#     # split rep into flexion,  extension, and rest (middle 1 sec prior to flexion)
#     rep_df_flx, rep_df_ext = split_rep_to_flex_ext(rep_df, rep_st, rep_et, flx_dur, ext_dur, rest_dur)
#     _,_ ,rep_df_rest = extract_rest_for_rep(df, trig_df, task_id, rep_id, n_reps_dict)

#     if combine_flx_ext:
#         # concatenate rep_df_flx and rep_df_ext
#         flx_ext_df = pd.concat([rep_df_flx, rep_df_ext], axis=0, ignore_index=True, sort=False)
#         # print(f"flx_ext_df.shape: {flx_ext_df.shape}  rep_df_flx:{rep_df_flx.shape}    rep_df_ext:{rep_df_ext.shape}")

#         # get start and end times of concatenated df
#         comb_st, comb_end = get_start_end_time(flx_ext_df)
#         bin_range_flx_ext = np.arange(comb_st, comb_end, bin_width)
#         bins_stat_comb = pd.DataFrame()

#     else:
#         # flex/ext start and end times
#         flx_st, flx_end = get_start_end_time(rep_df_flx)
#         ext_st, ext_end = get_start_end_time(rep_df_ext)
#         rest_st, rest_end = get_start_end_time(rep_df_rest)

    
#         bins_flx_stat = pd.DataFrame()
#         bins_ext_stat = pd.DataFrame()
#         bins_rest_stat = pd.DataFrame()

#         bin_range_flx = np.arange(flx_st, flx_end, bin_width)
#         bin_range_ext = np.arange(ext_st, ext_end, bin_width)
#         bin_range_rest = np.arange(rest_st, rest_end, bin_width)

#         # print(f"flx_start: {flx_st}  flx_end: {flx_end}  ext_start: {ext_st}  ext_end: {ext_end}")

#         # pack bin ranges into dict
#         bin_range_dict = {FLX_PHASE: bin_range_flx, EXT_PHASE: bin_range_ext, REST_PHASE: bin_range_rest}

#     # compute a statistics per channel over time bins of a specified width
#     # print(f"bin range:{bin_range_flx}")
#     for ch in range(N_CHANNELS):
     
#         if combine_flx_ext:
#             bin_comb_pow, _, _ = get_bin_stat(flx_ext_df[TIME_VAR], flx_ext_df[ch], bin_range_flx_ext, bin_stat, bin_width)
#             bins_stat_comb[ch] = bin_comb_pow

#         else:
#             # compute the mean power of each channel over time bins of a specified width        
#             bin_flx_pow,bin_flx_edge, _ = get_bin_stat(rep_df_flx[TIME_VAR], rep_df_flx[ch], bin_range_flx, bin_stat, bin_width)
#             bin_ext_pow, _, _ = get_bin_stat(rep_df_ext[TIME_VAR], rep_df_ext[ch], bin_range_ext, bin_stat, bin_width)
#             bin_rest_pow, _, _ = get_bin_stat(rep_df_rest[TIME_VAR], rep_df_rest[ch], bin_range_rest, bin_stat, bin_width)


#             assert bin_range_flx.shape[0] == bin_flx_edge.shape[0], "check bin_range_flx and bin_flx_edge"


#             # fill in the df columns
#             bins_flx_stat[ch] = bin_flx_pow
#             bins_ext_stat[ch] = bin_ext_pow
#             bins_rest_stat[ch] = bin_rest_pow

#             # print(f"bin_flx_edge:{bin_flx_edge}")
#     if combine_flx_ext:
#         bins_stat_comb.index = np.round(bin_range_flx_ext[:-1],2)

#         return bins_stat_comb, bin_range_flx_ext
#     else:
#         return bins_flx_stat, bins_ext_stat, bins_rest_stat, bin_range_dict


# TODO: remove not used
# def compute_stat_in_time_all(filt_df, trig_df_post, n_reps, sel_gest, flx_dur, ext_dur, rest_dur, bin_width, bin_stat):

#     bins_df_all = pd.DataFrame()
#     glob_rep_id = 0
#     for g, sel_gest_item in enumerate(sel_gest):
#         rep_count = n_reps[sel_gest_item.id]
#         g_phase = sel_gest_item.phase
#         g_id = sel_gest_item.id

#         for rep_id in range(rep_count):
    
#             bins_flx_stat, bins_ext_stat, _, bin_range_dict = compute_stat_in_time_for_rep_id(filt_df, trig_df_post, n_reps, rep_id, g_id, flx_dur, ext_dur, rest_dur, bin_width, bin_stat, combine_flx_ext=False)
#             # remove bad chs
#             bins_flx_stat = remove_bad_chs(bins_flx_stat)
#             bins_ext_stat = remove_bad_chs(bins_ext_stat)

#             # add label + append to df
#             if g_phase == FLX_PHASE:
#                 bins_flx_stat[REP_VAR] = glob_rep_id

#                 bins_flx_stat['label'] = g
#                 bins_flx_stat[TIME_VAR] = bin_range_dict[FLX_PHASE][:-1].round(2)
#                 # bins_df_all = bins_df_all.append(bins_flx_stat)

#                 bins_df_all = pd.concat([bins_df_all, bins_flx_stat], ignore_index=True)

#             elif g_phase == EXT_PHASE:
#                 bins_ext_stat[REP_VAR] = glob_rep_id
#                 bins_ext_stat['label'] = g
#                 bins_ext_stat[TIME_VAR] = bin_range_dict[EXT_PHASE][:-1].round(2)
#                 bins_df_all = pd.concat([bins_df_all, bins_ext_stat], ignore_index=True)

#                 # bins_df_all = bins_df_all.append(bins_ext_stat)
#             # print(bins_df_all.shape)
#             glob_rep_id += 1

#     return bins_df_all  


def extract_temporal_feat(eng_dataset, sel_gest, wind_size, stride, min_periods, feat='pow', remove_bad_chs=[]):

    avg_df_all = pd.DataFrame()
    glob_rep_id = 0
    for g, sel_gest_item in enumerate(sel_gest):
        rep_count = eng_dataset.task_rep_count[sel_gest_item.id]
        g_phase = sel_gest_item.phase
        g_id = sel_gest_item.id

        for rep_id in range(rep_count):
            flx_wind_avg_df, ext_wind_avg_df, _ = get_stat_moving_wind_for_rep_id(eng_dataset, rep_id, g_id, wind_size, stride, min_periods, 
                                                                                  feat, combine_flx_ext=False)

            # remove bad chs
            if len(remove_bad_chs) > 0:
                flx_wind_avg_df = remove_bad_chs(flx_wind_avg_df, remove_bad_chs)
                ext_wind_avg_df = remove_bad_chs(ext_wind_avg_df, remove_bad_chs)
            # flx_wind_avg_df = remove_bad_chs(flx_wind_avg_df)
            # ext_wind_avg_df = remove_bad_chs(ext_wind_avg_df)

            # add label + append to df
            if g_phase == FLX_PHASE:
                flx_wind_avg_df[REP_VAR] = glob_rep_id
                flx_wind_avg_df['label'] = g
                avg_df_all = pd.concat([avg_df_all, flx_wind_avg_df], ignore_index=True)

            elif g_phase == EXT_PHASE:
                ext_wind_avg_df[REP_VAR] = glob_rep_id
                ext_wind_avg_df['label'] = g
                avg_df_all = pd.concat([avg_df_all, ext_wind_avg_df], ignore_index=True)

            glob_rep_id += 1

    return avg_df_all  


def select_channels(df:DataFrame, chs_to_exclude:List):
    """
    Selects all channels except in chs_to_exclude list
    """
    return df[[col for col in df.columns if col not in chs_to_exclude]]



def extract_feat_for_task(eng_dataset:ENGDataset, task_id:int, feat:str='rms'):

    flex_stat_df = pd.DataFrame()
    ext_stat_df = pd.DataFrame()
    rest_stat_df = pd.DataFrame()


    for rep in range(eng_dataset.task_rep_count[task_id]):
        rep_df,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep, task_id)

        # split rep into flexion,  extension, and rest (middle 1 sec prior to flexion)
        rep_df_flx, rep_df_ext = split_rep_to_flex_ext(rep_df, rep_st, rep_et, eng_dataset)
        _,_ ,rep_df_rest = extract_rest_for_rep(eng_dataset, task_id, rep)
        logging.debug(f"rep:{rep}  task:{task_id}  flx:{rep_df_flx[TIME_VAR].iloc[-1]- rep_df_flx[TIME_VAR].iloc[0]} \next:{rep_df_ext[TIME_VAR].iloc[-1]- rep_df_ext[TIME_VAR].iloc[0]}")

        if feat=='rms':
            # compute the mean rms of each channel (across time)
            rep_df_flex_pow = rep_df_flx[rep_df_flx.columns[:-1]].pow(2).mean(axis=0)
            rep_df_ext_pow = rep_df_ext[rep_df_ext.columns[:-1]].pow(2).mean(axis=0)

            rep_df_flex_pow = rep_df_flex_pow.pow(0.5)
            rep_df_ext_pow = rep_df_ext_pow.pow(0.5)

            rep_df_rest_pow = rep_df_rest[rep_df_rest.columns[:-1]].pow(2).mean(axis=0)
            rep_df_rest_pow = rep_df_rest_pow.pow(0.5)

            
        elif feat == 'mav':
            rep_df_flex_pow = rep_df_flx[rep_df_flx.columns[:-1]].abs().mean(axis=0)
            rep_df_ext_pow = rep_df_ext[rep_df_ext.columns[:-1]].abs().mean(axis=0)
            rep_df_rest_pow = rep_df_rest[rep_df_rest.columns[:-1]].abs().mean(axis=0)
        elif feat == 'pow':
            rep_df_flex_pow = rep_df_flx[rep_df_flx.columns[:-1]].pow(2).mean(axis=0)
            rep_df_ext_pow = rep_df_ext[rep_df_ext.columns[:-1]].pow(2).mean(axis=0)
            rep_df_rest_pow = rep_df_rest[rep_df_rest.columns[:-1]].pow(2).mean(axis=0)
        flex_stat_df[rep] = rep_df_flex_pow
        ext_stat_df[rep] = rep_df_ext_pow
        rest_stat_df[rep] = rep_df_rest_pow

    return flex_stat_df, ext_stat_df, rest_stat_df
