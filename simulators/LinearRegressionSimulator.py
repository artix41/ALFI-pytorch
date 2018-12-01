from simulators import Simulator

import torch
from torch.distributions import Uniform, Normal
from math import pi
import torch.nn.functional as f


class LinearRegressionSimulator(Simulator):
    theta_dim = 2
    x_dim = 3

    def __init__(self, rand_seed=42, dim_x=1, angle_min=0., angle_max=pi/2, offset_min=-1., offset_max=1., sigma=.1, device="cpu"):
        super().__init__(rand_seed)
        self.p = dim_x
        self.angle_min = torch.tensor(angle_min).to(device)
        self.angle_max = torch.tensor(angle_max).to(device)
        self.offset_min = torch.tensor(offset_min).to(device)
        self.offset_max = torch.tensor(offset_max).to(device)
        self.theta_dim = dim_x + 1
        self.x_dim = torch.tensor(dim_x + 2).to(device)
        self.sigma = torch.tensor(sigma).to(device)
        self.device = device
        self.x_dim = dim_x + 2

    def get_data(self, n_exp, n_samp):
        thetas = self.get_thetas(n_exp)
        X_real = self.forward(thetas, n_samp)
        return thetas, X_real

    def get_thetas(self, n_exp):
        # Theta parametrized as an angle
        d = Uniform(self.angle_min, self.angle_max)
        thetas = d.sample(torch.Size([n_exp, self.p + 1]))
        thetas[thetas == 0] += 1e-4
        d = Uniform(self.offset_min, self.offset_max)
        thetas[:, -1] = d.sample(torch.Size([n_exp]))

        return thetas

    def forward(self, thetas, n_samp=1):
        thetas = thetas.clone()
        thetas[:, :-1] = torch.tan(thetas[:, :-1])

        n_exp = thetas.shape[0]

        d = Uniform(self.offset_min, self.offset_max)
        X = torch.cat((d.sample(torch.Size([n_exp, n_samp, self.p])), torch.ones(n_exp, n_samp, 1).to(self.device)), 2)
        Y = torch.matmul(X, thetas.unsqueeze(2)) + Normal(torch.tensor(0.).to(self.device), self.sigma).sample(torch.Size([n_exp, n_samp])).unsqueeze(2)

        X_real = torch.cat((X, Y), 2)
        return X_real

    def get_mle(self, samples):
        samples = samples.cpu()
        Y = samples[:, :, -1]
        X = samples[:, :, :-1]
        thetas = torch.zeros((X.shape[0], X.shape[2]))
        sigmas = torch.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            X_t = X[i, :, :]
            XT_t = X_t.transpose(0, 1)
            Y_t = Y[i, :]
            thetas[i, :] = torch.matmul(torch.matmul(torch.inverse(torch.matmul(XT_t, X_t)), XT_t), Y_t)
            thetas[i, :-1] = torch.atan(thetas[i, :-1])
            sigmas[i, :] = torch.diag(torch.matmul(XT_t, X_t)*self.sigma.cpu()/X.shape[1])
        return thetas, sigmas
