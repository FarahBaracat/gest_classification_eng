from constants import *
from utils.load_files import load_mat_data
import logging
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import enum
from collections import Counter

class Nerve(enum.Enum):
    MEDIAN_E1 = 'median_elec1'
    MEDIAN_E2 = 'median_elec2'
    ULNAR_E1 = 'ulnar_elec1'
    ULNAR_E2 = 'ulnar_elec2'
 


def get_nerve_ch_group() -> dict:
    """Returns a dictionary mapping electrode to channel group.

      """
    #CHECKTHIS: check the order of channels is correct: i.e that the first 14 belong to Median 1    
    n_chs_per_elec = 14
    grid_to_channel_group = {
                                Nerve.MEDIAN_E1: np.arange(n_chs_per_elec),  # grid 1
                                Nerve.MEDIAN_E2: np.arange(n_chs_per_elec, n_chs_per_elec*2, 1), # grid 2
                                Nerve.ULNAR_E1:np.arange(n_chs_per_elec*2, n_chs_per_elec*3, 1) ,  # grid 3
                                Nerve.ULNAR_E2: np.arange(n_chs_per_elec*3, n_chs_per_elec*4, 1)  # grid 4
                            }
    return grid_to_channel_group



@dataclass
class ENGDataset:
    # Setup information  
    n_channels:int = 56
    n_chs_per_elec:int = 14
    n_electrodes:int = 4
    n_elec_per_nerve:int = 2 # 2 electrodes are implanted per nerve: 2 in median and 2 in ulnar
    n_nerves:int = 2  # 2 nerves
    fs:int = 30000      # data sampling frequency 30kHz
    
    nerves_ch_group: dict = field(default_factory=get_nerve_ch_group ) # Keys are nerve and electrode number, values are list of channels on this electrode

    # Task information
    task_order: list[str] = field(default_factory=lambda: ['Tripod', 'ThOpp.', 'Power', 'UlnarFing.', 'FingAbd.'])
    max_rep_dur = 1 # in seconds
    flex_dur = 1   # in seconds
    ext_dur = 1    # in seconds
    rest_dur = 3  # in seconds

    # user-defined input
    day:int = 16        # day of recording
    session:str = '01'  # session of recording
    load_raw_data:bool = False  # boolean to either load the raw data as well or ignore 
    save_figs:bool = False  # boolean to save figures or not


    # data paths and filenames
    root_data_dir  = "/Users/farahbaracat/Library/CloudStorage/OneDrive-UniversitätZürichUZH/ENG upper dataset/Data_TIME_Marina"
    

    # data variables
    def __post_init__(self):
        self.n_tasks = self._get_n_tasks()
        if self.day==23:
            if self.session == '01':
                session = '03'
            if self.session == '02':
                session = '05'
            self.raw_data_file = f"AM_prese_{self.day}{session}_raw_ENG.mat"   
            self.post_data_file = f"AM_prese_{self.day}{session}_raw_ENG_ok.mat"
            self.raw_data_path = os.path.join(self.root_data_dir, f"day{self.day}_{self.session}",self.raw_data_file)
            self.post_data_path = os.path.join(self.root_data_dir, f"day{self.day}_{self.session}", self.post_data_file)

        else:
         
            self.raw_data_file = f"AM_prese_{self.day}{self.session}_raw_ENG.mat"
            self.post_data_file = f"AM_prese_{self.day}{self.session}_raw_ENG_ok.mat"
            self.raw_data_path = os.path.join(self.root_data_dir, f"day{self.day}",self.raw_data_file)
            self.post_data_path = os.path.join(self.root_data_dir, f"day{self.day}", self.post_data_file)
        
        if self.load_raw_data:
            self.raw_data = load_mat_data(self.raw_data_path)
        else:
            self.raw_data = None

        self.post_data = load_mat_data(self.post_data_path)
        self.task_rep_count = self._create_dict_of_reps()
        self._set_time_column()
        self.trig_df = self._organize_trigger_data()
        self.post_data_df = self._post_data_to_df()
        # else:
            # self.raw_data = None
            # self.post_data = None
            # self.task_rep_count = None
            # self.trig_df = None
            # self.post_data_df = None
        self.filt_df = pd.DataFrame()  # initialize an empty dataframe for filtered data
        self.filt_pipeline = None      # a dict with the filtering pipeline

    def _get_n_tasks(self):
        return len(self.task_order)
    

    def _create_dict_of_reps(self):
        task_rep_count = {}
        for task in range(len(self.post_data[SEG_VAR])):
            task_rep_count[task] = self.post_data[SEG_VAR][task].shape[0]
        logging.info(f"Creating dictionary of reps count per task: {task_rep_count}")
        return task_rep_count


    def _organize_trigger_data(self):
        """ Set task_id column based on the trigger info in post_data"""
        trig_df = pd.DataFrame(self.post_data[TRIG_VAR], columns=[TRIG_VAR])

        # add task id and match to length of trig_df
        task_id = np.repeat(np.arange(N_TASKS), list(self.task_rep_count.values()))
        trig_df[TASK_ID_COL] = task_id

        # assert that time difference between triggers is at least 3 seconds: 1 sec for flex, 1 sec for ext, 1 sec for rest
        min_rep_dur = 3    # in sec
        incomp_rep = trig_df[trig_df[TRIG_VAR].diff()< min_rep_dur].index

        if incomp_rep.any():
            prev_trig = [] # get the previous trigger to the incomplete reps
            for i in list(incomp_rep):
                prev_trig.append(i-1)
            print(f"Check on these triggers, they don't fulfill min rep criteria\n{trig_df.iloc[prev_trig + list(incomp_rep)]}")
            incomp_rep_task = trig_df.iloc[incomp_rep][TASK_ID_COL].values

            # Update the task_rep_count
            for task in incomp_rep_task:
                self.task_rep_count[task] -= dict(Counter(incomp_rep_task))[task]

            # Remove the incomplete reps
            trig_df.drop(incomp_rep, inplace=True)  

        return trig_df

    def _set_time_column(self):
        """ Extracts time column from the raw data since post data has no time column"""
        # Update: there is no need to use self.raw_data[TIME_VAR] since the time column can be aranged
        time_cut = np.arange(0, self.post_data[ENG_RAW_VAR].shape[1]/self.fs, 1/self.fs) #self.raw_data[TIME_VAR][:self.post_data[ENG_RAW_VAR].shape[1]]
        self.post_data[TIME_VAR] = time_cut
        logging.info(
            f"Time column of post_data{self.post_data[TIME_VAR].shape} \nRec column of post_data{self.post_data[ENG_RAW_VAR].shape}")


    def _post_data_to_df(self):
        data_df = pd.DataFrame(self.post_data[ENG_RAW_VAR].T)
        data_df[TIME_VAR] = self.post_data[TIME_VAR]
        return data_df

    def _detete_raw_data(self):
        del self.raw_data
        return None

