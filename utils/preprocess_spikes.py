from constants import N_REPS_PER_TASK, MAX_REP_DUR
import wandb
import torch
import numpy as np
import logging
from scipy import stats
import warnings


def dense_to_sparse(spikes, sel_gest, num_steps, num_inputs, rep_dur, dt, device, dtype):
    """
    Convert lists of spike times and ids into sparse tensors of shape (n_samples, num_steps, num_inputs)
    with 1s at spike times and 0s elsewhere.
    """

    # Prepare the dataset: sparse array at each dt:each sample should be (timestep, neuron_id)?
    n_samples = np.sum([N_REPS_PER_TASK[gest.id] for gest in sel_gest])   # n_rows in dataset

    input = torch.zeros((n_samples, num_steps, num_inputs), dtype=dtype, device=device)
    labels = torch.zeros((n_samples), dtype=dtype, device=device)

    # sparse representation
    row = 0

    row_input = 0  # row in input tensor
    for g, sel_gest_item in enumerate(sel_gest):
        rep_count = N_REPS_PER_TASK[sel_gest_item.id]
        g_phase = sel_gest_item.phase

        for rep_id in range(rep_count):
            sp_times, sp_ids = spikes[g].get_sp_for_rep(rep_id, g_phase)
            bin_range = np.arange(0, rep_dur + dt, dt)

            # mask different channels sequentially to fill in the input tensor
            n_neurons = np.unique(sp_ids)
            logging.debug(
                f"g:{g}, rep_id:{rep_id}, n_neurons:{n_neurons}\nunique:{np.unique(sp_ids)}\n")
            for neu in n_neurons:
                mask = np.array(sp_ids) == neu
                neu_sp_times = np.array(sp_times)[mask]
                logging.debug(neu, neu_sp_times)

                if len(neu_sp_times) > 0:
                    bin_stat, _, _ = stats.binned_statistic(
                        neu_sp_times, values=neu_sp_times, bins=bin_range, statistic='count')
                    input[row_input, :, neu] = torch.from_numpy(bin_stat).to(device)
            row_input += 1
            labels[row] = g
            row += 1
    
    # split the input tensor vertically (ie with respect to time)
    # if wandb.config['split_rep']:
        # n_splits = int(MAX_REP_DUR/split_bin)
        # input = torch.split(input, int(num_steps/n_splits), dim=1)
        # logging.info(f"input split into {n_splits} parts each:{split_bin} ms")

        # # concatenate all splits
        # input = torch.cat(input, dim=0)

        # # repeat labels accordingly
        # labels = labels.repeat(n_splits)
        # logging.info(f"\nInput size:{input.size()}\nLabels size:{labels.size()}\n")
        # num_steps = int(num_steps/n_splits)


    # raise warning that with the choice of dt, there is only one spike per bin
    if torch.sum(input > 1) == 0:
        warnings.warn("More than one spike in a bin. Choose a smaller dt.\n------------------\n")
    # assert torch.sum(input > 1) == 0, "More than one spike in a bin. Choose a smaller dt."
    return input, labels, num_steps


def split_rep_tensor_into_bins(input,labels,num_steps, bin_width):
    """
    Splits a single repetition into smaller time bins.
    Case1: non-overlapping windows. In this case, it slices the tensor vertically (since columns are time steps)
    Todo: Case2: overlapping windows...
    """

    if wandb.config['split_type'] == 'no_overlap':
        n_splits = int(MAX_REP_DUR/bin_width)
        input = torch.split(input, int(num_steps/n_splits), dim=1)
        logging.info(f"input split into {n_splits} parts each:{bin_width} ms")

        # concatenate all splits
        input = torch.cat(input, dim=0)

        # repeat labels accordingly
        labels = labels.repeat(n_splits)
        logging.info(f"\nInput size:{input.size()}\nLabels size:{labels.size()}\n")
        num_steps = int(num_steps/n_splits)

        # now the trials of each label are not consecutive anymore
        sorted_labels, sorted_ids = torch.sort(labels)
        sorted_input = input[sorted_ids]

    if wandb.config['split_type'] == 'with_overlap':
    
        bin_width_in_time_steps = int(bin_width / wandb.config['dt'])
        stride_in_time_steps =  int(bin_width_in_time_steps - wandb.config['overlap_perc'] * bin_width_in_time_steps) # in time step

        print(f"\nOverlap percentage: {wandb.config['overlap_perc']}   win_size:{bin_width} (in steps:{bin_width_in_time_steps})  stride:{stride_in_time_steps} time steps")
        print("------------------------------------")
        min_periods = 0 
        wins= [input[:,i:i+bin_width_in_time_steps]  for i in range(0, input.size(1), stride_in_time_steps)  
               if i+bin_width_in_time_steps <= input.size(1)+ min_periods]

        # Todo: split the labels accordingly and sort both input and labels.
        n_splits = len(wins) # number of overlapping windows
        num_steps= bin_width_in_time_steps
        wins_tensor = torch.cat(wins, dim=0)
        labels_exp = labels.repeat(n_splits)
#
        sorted_labels, sorted_ids = torch.sort(labels_exp)
        sorted_input = wins_tensor[sorted_ids]


    return sorted_input, sorted_labels, num_steps, n_splits
