import torch
import torch.nn as nn
import torch.nn.functional as f

class GRU(nn.Module):
    def __init__(self, input_size, st_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.st_size = st_size

        self.reset_gate = nn.Linear(input_size + st_size, st_size)
        self.update_gate = nn.Linear(input_size + st_size, st_size)
        self.out_gate = nn.Linear(input_size + st_size, st_size)
        
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)

    def forward(self, xt, st):
        stacked_inputs = torch.cat([xt, st], dim=1)

        update = f.sigmoid(self.update_gate(stacked_inputs))
        reset = f.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = f.tanh(self.out_gate(torch.cat([xt, st * reset], dim=1)))

        new_st = st * (1 - update) + out_inputs * update

        return new_st
