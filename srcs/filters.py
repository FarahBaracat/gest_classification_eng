
from scipy import signal
import numpy as np


class Filters:
    def __init__(self, prepipeline, samp_freq, n_notch=1):
        self.prepipeline = prepipeline
        self.samp_freq = samp_freq
        self.n_notch = n_notch

        self.notch_filter = self.create_notch_filter()
        self.mult_notch_filter = self.create_mult_notch_filter()
        if 'hp_order' in prepipeline:
            self.hp_filter = self.create_hp_filter()
        if 'lp_order' in prepipeline:
            self.lp_filter = self.create_lp_filter()
        if 'bp_order' in prepipeline:
            self.bp_filter = self.create_bp_filter()

    def create_notch_filter(self):
        w0 = self.prepipeline['notch_reject']
        q = self.prepipeline['notch_reject'] / \
            self.prepipeline['notch_bandwidth']
        b, a = signal.iirnotch(w0, q, self.samp_freq)
        return b, a

    def create_mult_notch_filter(self):
        w0 = self.prepipeline['notch_reject']
        q = self.prepipeline['notch_reject'] / \
            self.prepipeline['notch_bandwidth']  # quality factor
        coef_tuples = [(signal.iirnotch(w0 * i, q, self.samp_freq))
                       for i in range(1, self.n_notch)]
        return coef_tuples

    def create_hp_filter(self):
        # normalize cutoff frequency
        w_hp = self.prepipeline['hp_cutoff_freq'] / (self.samp_freq / 2)
        b_hp, a_hp = signal.butter(
            self.prepipeline['hp_order'], w_hp, btype='highpass')
        return b_hp, a_hp

    def create_bp_filter(self):
        bp_order, bp_cutoff_freq = self.prepipeline['bp_order'], self.prepipeline['bp_cutoff_freq']
        b_bp, a_bp = signal.butter(
            bp_order, bp_cutoff_freq / (self.samp_freq / 2), btype='bandpass')
        return b_bp, a_bp

    def create_lp_filter(self):
        downsample_freq = int(
            self.samp_freq / self.prepipeline['downsample_fact'])
        b_coef = np.ones((int(downsample_freq * 0.1),)) / \
            (downsample_freq * 0.1)

        return b_coef





