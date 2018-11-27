from simulators import Simulator

import torch
import numpy as np
from torch.distributions import Uniform, Poisson

class PoissonSimulator(Simulator):
    theta_dim = 1
    x_dim = 1

    def __init__(self, rand_seed=42, device="cpu", min=.2, max=7.):
        super().__init__(rand_seed)
        self.device = device
        self.min = torch.tensor(min).to(device)
        self.max = torch.tensor(max).to(device)

    def get_data(self, n_exp, n_samp):
        #d = Uniform(self.min, self.max)
        #thetas = d.sample(torch.Size([n_exp]))
        #X_real = Poisson(torch.exp(thetas)).sample(torch.Size([n_samp, 1])).permute([2, 0, 1])
        thetas = self.get_thetas(n_exp)
        X_real = self.forward(thetas, n_samp)
        return thetas.reshape([-1, 1]), X_real

    def get_thetas(self, n_exp):
        d = Uniform(self.min, self.max)
        return d.sample(torch.Size([n_exp, 1]))

    def forward(self, theta, n_samp=1):
        return torch.distributions.poisson.Poisson(torch.exp(theta.view(theta.shape[0], -1)))\
            .sample(torch.Size([n_samp])).permute(1, 0, 2)

    def get_mle(self, samples):
        return torch.log(samples.mean(1)), 0*samples.std(1)/np.sqrt(samples.shape[1])
