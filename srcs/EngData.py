from constants import *
# from srcs.Filters import Filters, filter_ch
import logging
import mat73
import os
import pandas as pd
import numpy as np


class EngData():
    def __init__(self, day, session, data_dir=DATA_DIR):
        self.day = day
        self.session = session
        self.data_dir = data_dir

        self.raw_data_file = f"AM_prese_{self.day}{self.session}_raw_ENG.mat"
        self.post_data_file = f"AM_prese_{self.day}{self.session}_raw_ENG_ok.mat"
        self.raw_data_path = os.path.join(self.data_dir, self.raw_data_file)
        self.post_data_path = os.path.join(self.data_dir, self.post_data_file)

        self.raw_data = load_mat_data(self.raw_data_path)
        self.post_data = load_mat_data(self.post_data_path)

        # Organize data in dataframes
        self.task_rep_count = self._create_dict_of_reps()
        self.trigger = self._organize_trigger_data()  # from post-processed data

    def _organize_trigger_data(self):
        trig_df = pd.DataFrame(self.post_data[TRIG_VAR], columns=[TRIG_VAR])

        # add task id and match to length of trig_df
        task_id = np.repeat(np.arange(N_TASKS), list(self.task_rep_count.values()))
        trig_df['task_id'] = task_id
        return trig_df

    def _create_dict_of_reps(self):
        task_rep_count = {}
        for task in range(len(self.post_data[SEG_VAR])):
            task_rep_count[task] = self.post_data[SEG_VAR][task].shape[0]
        logging.info(f"Creating dictionary of reps count per task: {task_rep_count}")

        return task_rep_count

    def set_time_column(self):
        """ Extracts time column from the raw data since post data has no time column"""
        time_cut = self.raw_data[TIME_VAR][:self.post_data[ENG_RAW_VAR].shape[1]]
        time_cut, time_cut[-1]
        self.post_data[TIME_VAR] = time_cut
        logging.info(
            f"Time column of post_data{self.post_data[TIME_VAR].shape} \nRec column of post_data{self.post_data[ENG_RAW_VAR].shape}")

    def to_dataframe(self):
        data_df = pd.DataFrame(self.post_data[ENG_RAW_VAR].T)
        data_df[TIME_VAR] = self.post_data[TIME_VAR]
        return data_df


# def apply_filter_pipeline(pipeline, data_df, n_ch=N_CHANNELS):

#     if 'notch_reject' not in pipeline:
#         raise ValueError("Notch reject frequency not specified in pipeline")

#     hi_band_freq = pipeline['bp_cutoff_freq'][-1]
#     print(f"Applying filter to max freq:{hi_band_freq} Hz")
#     n_notch = int(hi_band_freq / pipeline['notch_reject'])
#     n_samples = data_df.shape[0]

#     filters = Filters(pipeline, ENG_FS, n_notch)

#     # notch filter parameters
#     notch_filters = filters.create_mult_notch_filter()

#     # bandpass filer parameters
#     b_bp, a_bp = filters.create_bp_filter()

#     notch_filt_data = np.zeros((n_samples, n_ch))
#     bp_filt_data = np.zeros((n_samples, n_ch))
#     for ch in range(n_ch):
#         raw_ch = np.array(data_df[ch])
#         print(f"Notch filter ch#{ch} shape:{raw_ch.shape[0]}")
#         for j, (b_notch, a_notch) in enumerate(notch_filters):
#             if j == 0:
#                 raw_notch = filter_ch(raw_ch, b_notch, a_notch)
#             else:
#                 raw_notch = filter_ch(raw_notch, b_notch, a_notch)
#         notch_filt_data[:, ch] = raw_notch

#         print(f"BPF ch#{ch} {pipeline['bp_cutoff_freq']}")
#         bp_filt_data[:, ch] = filter_ch(notch_filt_data[:, ch], b_bp, a_bp)

    # # Organize filtered data in dataframe
    # filt_df = pd.DataFrame(bp_filt_data)
    # filt_df[TIME_VAR] = eng_df[TIME_VAR]

    # return notch_filt_data, bp_filt_data


def load_mat_data(data_path):
    logging.info(f"Loading data from {data_path}")
    return mat73.loadmat(data_path)
