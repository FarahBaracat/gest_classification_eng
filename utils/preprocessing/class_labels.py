from typing import List, Dict
from srcs.engdataset import ENGDataset
from collections import namedtuple

Gest_namedtup = namedtuple('gesture', ['id', 'phase'])


def reverse_labels_map(labels_map: Dict[int, str]):
    """
    Reverses the labels map to have the class label as the key and the gesture as the value.

    """
    rev_labels_map = {}
    for k, v in labels_map.items():
        rev_labels_map[v] = k
    return rev_labels_map

def encode_gest_phase(eng_dataset: ENGDataset, sel_gest_phase: List[Gest_namedtup], 
                      labels_map: Dict[int, str]):
    """
    Encodes class labels from the selected gestures considering the phase as well.

    """
    enc_class_ids = []
    enc_class_labels = []

    rev_labels_map = reverse_labels_map(labels_map)

    for sel_gest_item in sel_gest_phase:
        g = eng_dataset.task_order[sel_gest_item.id]            
        phase = sel_gest_item.phase
        print(f"Gesture id:{sel_gest_item.id}  task:{g}  phase:{phase}")

        gest_phase = f"{g} {phase}" 
        if g == 'FingAbd.':
            gest_phase = 'FingAbd.' if phase=='Close' else 'FingAdd.'
        
        # look for this g in the labels_map values
        if gest_phase not in rev_labels_map.keys():
            raise ValueError(f"Gesture {gest_phase} not found in the labels map")
        enc_class_ids.append(rev_labels_map[gest_phase])
        enc_class_labels.append(gest_phase)

    return enc_class_ids, enc_class_labels
