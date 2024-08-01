from scipy.fft import rfft, rfftfreq


def get_fft(ch_signal, fs_eng):
    """
    Filtering with rfft instead of fft
    "When the DFT is computed for purely real input, the output is Hermitian-symmetric, i.e. the negative frequency 
    terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency 
    terms are therefore redundant. 
    This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore n//2 + 1."
    """
    # Number of samples in normalized_tone
    N = ch_signal.shape[0]  # patient_df.shape[0]

    # yf = fft(np.array(patient_df[sel_ch[0]]))
    yf = rfft(ch_signal)  # using rfft since this is a purely real input
    xf = rfftfreq(N, 1 / fs_eng)
    return xf, yf
