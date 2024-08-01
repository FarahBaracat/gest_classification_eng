
from srcs.engdataset import ENGDataset
from constants import *

from typing import Dict
import pandas as pd
import logging

def get_trig_start_end_for_task(df:pd.DataFrame, trig_df:pd.DataFrame, 
                                rep_id:int, task_id:int, 
                                n_reps_dict:Dict[int,int]):
    """
    Gets the start and end triggers for a given task_id and rep_id
    """
    is_last = False

    if rep_id == n_reps_dict[task_id] -1 :
        is_last = True

    # get the start and end time of the rep
    rep_start = trig_df[trig_df[TASK_ID_COL] == task_id][TRIG_VAR].iloc[rep_id]
    
    # Last repetition needs special handling
    if is_last:
        if task_id == N_TASKS-1: # if last rep and task then crop till end of recording
            rep_end = df[TIME_VAR].iloc[-1]
        else: # if last rep only, then the end matches the start of the following task
            rep_end = trig_df[trig_df[TASK_ID_COL] == task_id+1][TRIG_VAR].iloc[0]
    else:
        rep_end = trig_df[trig_df[TASK_ID_COL] == task_id][TRIG_VAR].iloc[rep_id + 1]
    return rep_start,rep_end



def extract_rest_for_rep(eng_dataset:ENGDataset, task_id:int, rep_id:int):
    """
    Extract the rest period for a given rep. This is the middle 1 sec before the start of the repetiton (flexion part).
    Each repetition is preceded by 3 secs of rest. Here, I extract the middle 1 sec of rest.
    """
    n_reps_dict =eng_dataset.task_rep_count
    df = eng_dataset.filt_df
    if rep_id ==0:
        if task_id ==0:
            _,rep_st, rep_et = get_single_rep_for_task(eng_dataset,rep_id, task_id)
            rest_start = rep_st - 2 # remove 2 seconds from start of first rep
            rest_end = rest_start + 1 # get only 1 sec of rest

        else:
            _,rep_st, rep_et = get_single_rep_for_task(eng_dataset, rep_id=n_reps_dict[task_id-1]-1, task_id=task_id-1)
            rest_start = rep_et - 2 # remove 2 seconds from start of first rep
            rest_end = rest_start + 1 # get only 1 sec of rest
    else:
        _,rep_st, rep_et = get_single_rep_for_task(eng_dataset,rep_id, task_id)
        rest_start = rep_st - 2 # remove 2 seconds from start of first rep
        rest_end = rest_start + 1 # get only 1 sec of rest
    
    # slice df
    rest_df = df[(df[TIME_VAR] >= rest_start) & (df[TIME_VAR] < rest_end)]
    
    return rest_start, rest_end,rest_df



def get_single_rep_for_task(eng_dataset:ENGDataset, rep_id:int, task_id:int):
    """
    Each rep is 5 sec, starts with flexion for 1 sec, followed by extension 1 sec and then 3 sec of rest
    """
    # check is_last correct
    # assert rep_id < n_reps[task_id], f"rep_id {rep_id} is larger than n_reps {n_reps[task_id]} for task {task_id}"
    trig_df = eng_dataset.trig_df
    df = eng_dataset.filt_df
    task_rep_count = eng_dataset.task_rep_count
    rep_start, rep_end = get_trig_start_end_for_task(df, trig_df, rep_id, task_id, task_rep_count)

        
    # slice the rep
    rep_df = df[(df[TIME_VAR] >= rep_start) & (df[TIME_VAR] < rep_end)]
    logging.debug(f"rep {rep_id} start: {rep_start}, end: {rep_end}  shape:{rep_df.shape}")

    return rep_df, rep_start, rep_end

