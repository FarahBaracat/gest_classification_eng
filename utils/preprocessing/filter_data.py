import numpy as np
from scipy import signal

from srcs.filters import Filters




def filter_ch(ch_data, b_coef, a_coef):
    return signal.filtfilt(b_coef, a_coef, ch_data)



def apply_filter_pipeline(eng_dataset):

    data_df = eng_dataset.post_data_df
    pipeline = eng_dataset.filt_pipeline

    if 'notch_reject' not in pipeline:
        raise ValueError("Notch reject frequency not specified in pipeline")

    hi_band_freq = pipeline['bp_cutoff_freq'][-1]
    print(f"Applying filter to max freq:{hi_band_freq} Hz")
    n_notch = int(hi_band_freq / pipeline['notch_reject'])
    n_samples = data_df.shape[0]

    filters = Filters(pipeline, eng_dataset.fs, n_notch)

    # notch filter parameters
    notch_filters = filters.create_mult_notch_filter()

    # bandpass filer parameters
    b_bp, a_bp = filters.create_bp_filter()

    notch_filt_data = np.zeros((n_samples, eng_dataset.n_channels))
    bp_filt_data = np.zeros((n_samples, eng_dataset.n_channels))
    for ch in range(eng_dataset.n_channels):
        raw_ch = np.array(data_df[ch])
        print(f"Notch filter ch#{ch} shape:{raw_ch.shape[0]}")
        for j, (b_notch, a_notch) in enumerate(notch_filters):
            if j == 0:
                raw_notch = filter_ch(raw_ch, b_notch, a_notch)
            else:
                raw_notch = filter_ch(raw_notch, b_notch, a_notch)
        notch_filt_data[:, ch] = raw_notch

        print(f"BPF ch#{ch} {pipeline['bp_cutoff_freq']}")
        bp_filt_data[:, ch] = filter_ch(notch_filt_data[:, ch], b_bp, a_bp)

    # TODO: Organize filtered data in dataframe
    # filt_df = pd.DataFrame(bp_filt_data)
    # filt_df[TIME_VAR] = eng_df[TIME_VAR]

    return notch_filt_data, bp_filt_data
