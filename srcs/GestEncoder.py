import copy
import numpy as np


class GestSpEncoder(object):
    """Helper class to organize the encoded data into spikes"""

    def __init__(self, spikes, gest_label, label_to_gest):
        """
        spikes: a dicr of spike data for each repetition. Key is the rep id, value is a list of spikes"""
        self.spikes = spikes
        self.gest_label = gest_label
        self.gest_name = label_to_gest[gest_label]

    def get_sp_for_rep(self, rep_id, phase):
        """Get the spike data for a given repetition and phase.
        Phase: can be "flx" or "ext" for flexion and extension, respectively."""
        return self.spikes[rep_id][phase]
