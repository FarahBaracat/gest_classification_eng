from srcs.engdataset import ENGDataset, Nerve

import numpy as np
import logging


def detect_bad_channels(eng_dataset:ENGDataset, std_threshold:float):
    """
    Flag bad channels based on the channel standard deviation compared to other channels in the 
    same electrode. I use the filtered dataframe.
    from SpikeInterface: If the standard deviation of a channel is greater than std_threshold 
    times the median of all channels standard deviations, the channel is flagged as noisy.


    """
    # Loop per electrode: compute the median of all channels stds in the given electrode
    bad_channels = []
    bad_channels_std = []
    for  nerve_elec in list(Nerve): 
        # Get all channels in the corresponding electrode
        elec_chs = eng_dataset.nerves_ch_group[nerve_elec]
        elec_chs_std = np.std(eng_dataset.filt_df[elec_chs], axis=0)
        elec_chs_std_median = np.median(elec_chs_std)
        logging.info(f"{nerve_elec}: {elec_chs}\nChannels median of stds: {elec_chs_std_median}\n")
        
        bad_ch_mask = elec_chs_std[elec_chs_std >= std_threshold * elec_chs_std_median]
        bad_channels.extend(list(bad_ch_mask.index))
        bad_channels_std.extend(list(bad_ch_mask.values))
    return bad_channels, bad_channels_std