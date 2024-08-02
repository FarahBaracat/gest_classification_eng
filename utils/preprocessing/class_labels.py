from typing import List, Dict
from srcs.engdataset import ENGDataset
from collections import namedtuple

Gest_namedtup = namedtuple('gesture', ['id', 'phase'])


def encode_gest_phase(eng_dataset: ENGDataset, sel_gest_phase: List[Gest_namedtup], 
                      labels_map: Dict[int, str]):
    """
    Encodes class labels from the selected gestures considering the phase as well.

    """
    enc_classes = []
    enc_class_labels = []
    for sel_gest_item in sel_gest_phase:
        g = eng_dataset.task_order[sel_gest_item.id]
        phase = sel_gest_item.phase
        print(sel_gest_item.id, g)
        # look for this g in the labels_map values
        for k, v in labels_map.items():
            if f"{g} {phase}" in v:
                enc_classes.append(k)
                enc_class_labels.append(v)
    return enc_classes, enc_class_labels
