import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from models import GRU


class RIM(nn.Module):
                          
    def __init__(self, input_size, st_size, hidden_size, output_size, bounded=-1, lr=.001):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.st_size = st_size
        self.lr = lr

        self.fc0_layer = nn.Linear(input_size, hidden_size)
        self.fc1_layer = nn.Linear(hidden_size, hidden_size)
        self.rnn_layer = GRU(hidden_size, st_size)
        self.fc3_layer = nn.Linear(st_size, hidden_size)
        self.fc4_layer = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.bounded = bounded

    def forward(self, xt, st):
        out = f.relu(self.fc0_layer.forward(xt))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        out = f.relu(self.fc1_layer.forward(out))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        st_out = self.rnn_layer.forward(out, st)
        out = f.relu(self.fc3_layer.forward(st_out))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        if self.bounded > 0:
            out = torch.clamp(self.fc4_layer.forward(out), -self.bounded, self.bounded)
        else:
            out = self.fc4_layer.forward(out)

        return out, st_out

    def backprop(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def loss(self, theta, list_psi_t):
        loss_t = self.loss_func(theta, list_psi_t)
        return self.weight_func(loss_t)

    def init_hidden(self, batch_dim=1):
        return torch.zeros((batch_dim, self.st_size))
