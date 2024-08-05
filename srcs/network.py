# create dataset
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset
from snntorch import functional as SF
import snntorch as snn

from snntorch import surrogate
from snntorch import functional as SF


torch.manual_seed(52)



class EMGCustomDataset(Dataset):
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def __getitem__(self, idx):
        label = self.labels[idx]
        input = self.input[idx]
        return input, label

    def __len__(self):
        return len(self.labels)


class Net(nn.Module):
    def __init__(self, num_steps, num_inputs, num_outputs):
        super().__init__()
        self.spike_grad = surrogate.fast_sigmoid()
        self.num_steps = num_steps
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_outputs)
        # Synaptic temporal dynamics
        alpha = torch.exp(-torch.tensor(wandb.config['dt'])/torch.tensor(wandb.config['tau_syn']))   # synaptic current decau rate
        beta = torch.exp(-torch.tensor(wandb.config['dt'])/torch.tensor(wandb.config['tau_mem']))


        if wandb.config['learn_tausyn']:
            alpha_dist = torch.rand(num_outputs)
            nn.init.normal_(alpha_dist, mean=alpha, std=0.05 * alpha)
        else:
            alpha_dist = alpha
        
        if wandb.config['learn_taumem']:
            beta_dist = torch.rand(num_outputs)
            nn.init.normal_(beta_dist, mean=beta, std=0.05 * beta)
        else:
            beta_dist = beta

        if wandb.config['lif_type'] == 'synaptic':
            self.lif1 = snn.Synaptic(beta=beta_dist, alpha = alpha_dist, spike_grad=self.spike_grad, reset_mechanism='subtract', 
                                     threshold=torch.ones(num_outputs),#1.0, #torch.ones(num_outputs),
                                     learn_threshold=True,
                                     learn_alpha=wandb.config['learn_tausyn'],
                                     learn_beta=wandb.config['learn_taumem'])
        else:
            self.lif1 = snn.Leaky(beta=beta, spike_grad=self.spike_grad, reset_mechanism='zero')

        if wandb.config['w_init_dist'] == 'uniform':
            nn.init.uniform_(self.fc1.weight, a=wandb.config['w_init_a'], b=wandb.config['w_init_b'])
        elif wandb.config['w_init_dist'] == 'normal':
            nn.init.normal_(self.fc1.weight, mean=wandb.config['w_init_mean'],
                             std=0.1*wandb.config['w_init_mean'])

        # nn.init.constant_(self.fc1.bias, 0.0)
        # Todo: add empty tensors to record spikes and membrane potentials

    def forward(self, spk_in):
        # Record the final layer
        rec = {'spk1': [], 'mem1': [] ,'cur1': [], 'spk_in': []}
        # Initialize hidden states at t=0

        if wandb.config['lif_type'] == 'synaptic':
            syn1, mem1 = self.lif1.init_synaptic()
            rec['syn1'] = []
        else:
            mem1 = self.lif1.init_leaky()


        
       

        for step in range(self.num_steps):  # neuron simulation
            # post-synaptic current <-- spk_in x weight
            cur1 = self.fc1(spk_in[:, step])
            # mem[t+1] <--post-syn current + decayed membrane

            if wandb.config['lif_type'] == 'synaptic':
                spk1,syn1, mem1 = self.lif1(cur1, syn1,mem1)
                rec['syn1'].append(syn1)

            else:
                spk1, mem1 = self.lif1(cur1, mem1)


            # clip mem1 to 0
            mem1 = torch.clamp(mem1, min=0)

            # record the variables
            rec['spk_in'].append(spk_in[:,step])
            rec['cur1'].append(cur1)
            rec['spk1'].append(spk1)
            rec['mem1'].append(mem1)

        # stack the recorded variables
        for var in rec.keys():
            rec[var] = torch.stack(rec[var], dim=0)
        
        return rec
