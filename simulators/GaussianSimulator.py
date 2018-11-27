from simulators import Simulator

import torch
from torch.distributions import Uniform, Normal


class GaussianSimulator(Simulator):
    theta_dim = 2
    x_dim = 1

    def __init__(self, min=[-10.0, -2.0], max=[10.0, 1.5]):
        super().__init__()
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)

    def get_data(self, n_exp, n_samp):
        thetas = self.get_thetas(n_exp)
        X_real = self.forward(thetas, n_samp)
        return thetas, X_real

    def get_thetas(self, n_exp):
        d = Uniform(self.min, self.max)
        return d.sample(torch.Size([n_exp]))

    def forward(self, theta, n_samp=1):
        return Normal(theta[:, 0], torch.exp(theta[:, 1])).sample(torch.Size([n_samp, 1])).permute([2, 0, 1])

    def get_mle(self, samples):
        mu = samples.mean(1)
        sigma = torch.log(samples.std(1))
        thetas = torch.cat((mu, sigma), 1)
        return thetas, 0*thetas
