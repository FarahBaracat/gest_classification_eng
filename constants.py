import numpy as np
import os

ENG_FS = 30*1000  # 30kHz
N_CHANNELS = 56
N_CHS_ELEC = 14  # number of channels per electrode
N_ELECTRODES = 2  # number of electrodes per nerve

DAY = 16
SESSION = '01'

# Channels implanted in median and ulnar nerves
MEDIAN_CHS = np.arange(N_CHS_ELEC*N_ELECTRODES)
ULNAR_CHS = np.arange(
    MEDIAN_CHS[-1]+1, MEDIAN_CHS[-1]+1+N_CHS_ELEC*N_ELECTRODES, 1)

# ENG DATA VARIABLES
TRIG_VAR = 'Trigger'
ENG_RAW_VAR = 'REC'
TIME_VAR = 'Time'
SEG_VAR = 'SEGM_tot'
REP_VAR = 'rep_id'
LABEL_VAR = 'label'

# Dataframe column names
TASK_ID_COL = 'task_id'
REP_ID_COL = 'rep_id'
FEAT_WIN_COL = 'feature_window'
LABEL_COL = 'class_label'

# Number of tasks
N_TASKS = 5
TASK_ORDER = ['Tripod', 'ThOpp.', 'Power', 'UlnarFing.', 'FingAbd.']

CLASS_TO_GEST = {0:'Tripod Open',
                 1:'ThOpp. Close',
                 2: 'UlnarFing. Close',
                 3: 'FingAbd.'}


FLX_PHASE = 'Close'
EXT_PHASE = 'Open'
REST_PHASE = 'Rest'


# Figures styling
TIME_XLAB = 'time (s)'

# Number of reps per task
N_REPS_PER_TASK = {0: 10, 1: 10, 2: 10, 3: 7, 4: 10}
MAX_REP_DUR = 1  # in sec
FLX_DUR = 1 # in seconds
EXT_DUR = 1 # in seconds
REST_DUR = 3 # in seconds

# Data directory
DATA_DIR = "/Users/farahbaracat/Library/CloudStorage/OneDrive-UniversitätZürichUZH/ENG upper dataset/Data_TIME_Marina" #'../data' 
POST_PROC_DIR = '../data/post_proc'
FILTERED_DIR = os.path.join(POST_PROC_DIR,'filtered')
ENCODED_DIR = '../data/encoded'
CLF_RESULTS_DIR = '../data/clf_results'

# Figures directories
FIG_DIR = os.path.join('figures',f'day{DAY}{SESSION}' )
CLF_FIG = os.path.join(FIG_DIR, 'clf')
# DAY_SESS_FIG = f'day{DAY}{SESSION}'
PCA_FIG = os.path.join(FIG_DIR, 'pca')
SNN_FIG = os.path.join(FIG_DIR, 'snn')
LDA_FIG = os.path.join(FIG_DIR, 'lda')


# plotting palette
COLOR_DICT = {'pumpkin': '#d35400', 'midnight_blue': '#2c3e50', 'pomgrenate': '#c0392b', 'green_sea': '#16a085',
              'wisteria': '#8e44ad', 'orange': '#f39c12', 'clouds': '#7f8c8d', 'naval': '#40739e', 'purple': '#8c7ae6',
              'viz_violet': '#9d02d7', 'viz_orange':'#fa8775', 'viz_rose': '#ea5f94',
              'viz_blue':'#361AE5', 'belize':'#2980b9', 'dark_cyan':'#028189',
              'samaritan': '#3c6382', 'dupain':'#60a3bc'}
GEST_COLORS = [ COLOR_DICT['viz_orange'], COLOR_DICT['green_sea'], COLOR_DICT['wisteria'] , COLOR_DICT['naval']]
LEGEND_ALPHA = 0.8
XLAB_PAD = 5
YLAB_PAD = 5
ACC_LABEL = 'Balanced accuracy (%)'