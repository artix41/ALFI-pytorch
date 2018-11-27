import torch
import torch.nn as nn
import torch.nn.functional as f


class ThetaAggregator(nn.Module):
    def __init__(self, input_size, config, output_size, lr=.001):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.lr = lr

        self.fc1_layer = nn.Linear(input_size, config['hidden_size'])
        self.fc2_layer = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc4_layer = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc5_layer = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc3_layer = nn.Linear(config['hidden_size'], output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward_step(self, xt):
        out = f.relu(self.fc1_layer(xt))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        out = f.relu(self.fc2_layer(out))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        out = f.relu(self.fc4_layer(out))
        out = f.relu(self.fc5_layer(out))

        out = f.relu(self.fc3_layer(out))
        
        return out

    # X : b_size x mb_size x x_dim
    # Out : b_size x out_dim
    def forward(self, X, phase):
        self.training = True if phase=="train" else False
        b_size, mb_size,  x_dim = X.shape
        out = self.forward_step(X.view(-1, x_dim)).view(b_size, mb_size, -1)
        return out
