
from constants import TIME_VAR
from utils.preprocess_analog import get_single_rep_for_task
import copy
import numpy as np
import pandas as pd


def copy_spmon(encoded_sp):
    sp_times = copy.deepcopy(np.array(encoded_sp.t))
    sp_ids = copy.deepcopy(np.array(encoded_sp.i))
    return sp_times, sp_ids




def organize_tasks_spikes(gest_sp_dict):
    """Converts spike monitors to arrays of spike times and ids.
    gest_sp_dict: a dict (key = rep_id) of dict(key=phase) and value = spike monitor"""

    spikes = {}
    for rep_id in gest_sp_dict.keys():
        spikes[rep_id] = {}
        for phase in gest_sp_dict[rep_id].keys():
            sp_times, sp_ids = copy_spmon(gest_sp_dict[rep_id][phase])
            spikes[rep_id][phase] = (sp_times, sp_ids)
    return spikes



def prepare_data_to_encode(sel_gest,filt_df, trig_df_post,flx_dur, ext_dur, rest_dur,n_reps):
    """Selects data to encode based on the selected gestures"""

    cache_df = pd.DataFrame()

    for  gest_tuple in sel_gest:        
        reps_count = n_reps[gest_tuple.id]

        for rep_id in range(reps_count):
            rep_df, rep_st, rep_et = get_single_rep_for_task(filt_df, trig_df_post, rep_id, gest_tuple.id, task_rep_count=n_reps)
            rep_df_flx, rep_df_ext = segment_flx_ext(rep_df, rep_st, rep_et, flx_dur, ext_dur, rest_dur)


            if gest_tuple.phase == 'flx':
                signal = rep_df_flx
            elif gest_tuple.phase == 'ext':
                signal = rep_df_ext
                
        
            # Holding in the data I am converting to spikes
            tmp_df = np.abs(signal.drop([TIME_VAR], axis=1))
            tmp_df['max_val'] = np.abs(signal.drop([TIME_VAR], axis=1)).max().max()
            tmp_df['min_val'] = np.abs(signal.drop([TIME_VAR], axis=1)).min().min()

            tmp_df['rep_id'] = rep_id
            tmp_df['task_id'] = gest_tuple.id
            tmp_df['phase'] = gest_tuple.phase

            tmp_df[TIME_VAR] = signal[TIME_VAR]

            cache_df = pd.concat([cache_df,tmp_df], axis=0)

    return cache_df

