import pickle as pkl
import os
import logging
from constants import ENCODED_DIR
# from srcs.gest_sp_encoder import GestSpEncoder

import mat73



def load_mat_data(data_path):
    logging.info(f"Loading data from {data_path}")
    return mat73.loadmat(data_path)


def load_encoded_data(tau_mem, vth, sel_gest, file_prefix='engsquared_to_spikes', file_suffix=''):
    tuple_to_list = [
        f'{gest_tuple.id}_{gest_tuple.phase}'for gest_tuple in sel_gest]
    pkl_file = f'data/encoded/{file_prefix}_tau_{tau_mem}_vth_{vth}_{tuple_to_list}_{file_suffix}.pkl'
    # this is the file encoding MAV with 20 ms bin
    # pkl_file = f"data/encoded/{file_prefix}_tau_20. ms_vth_0.05_['0_ext', '1_flx', '3_flx', '4_flx'].pkl"
    if os.path.exists(pkl_file):
        logging.info(f'Loading encoded data from {pkl_file}')
        with open(pkl_file, 'rb') as f:
            return pkl.load(f)
    else:
        exit(f'File {pkl_file} does not exist. Run encode_data.py first.')
