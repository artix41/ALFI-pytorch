from simulators import Simulator

import torch
from torch.distributions import Uniform, Normal, Bernoulli, Exponential

class MultiDistriSimulator(Simulator):
    theta_dim = 2
    x_dim = 5
    R = torch.tensor([[ 1.31229955,  0.10499961,  0.48310515, -0.3249938,  -0.26387927],
                     [ 0.10499961,  1.15833058, -0.55865473,  0.25275522, -0.39790775],
                     [ 0.48310515, -0.55865473,  2.25874579, -0.52087938, -0.39271231],
                     [-0.3249938,   0.25275522, -0.52087938,  1.4034925,  -0.63521059],
                     [-0.26387927, -0.39790775, -0.39271231, -0.63521059,  1.        ]])

    def __init__(self, rand_seed=42, min=-6, max=-6):
        super().__init__(rand_seed)
        self.min = min
        self.max = max

    def get_data(self, n_exp, n_samp):
        d = Uniform(self.min, self.max)
        thetas = d.sample(torch.Size([n_exp, 2]))

        X = self.forward(thetas, n_samp=n_samp)

        return thetas.reshape([-1, 2]), X

    def get_thetas(self, n_exp):
        d = Uniform(self.min, self.max)
        return d.sample(torch.Size([n_exp, 2]))

    def forward(self, theta, n_samp=1):
        n_exp = theta.shape[0]
        n_samp = torch.Size([n_samp, 1])

        d0 = Normal(theta[:, 0], 1 * torch.ones(n_exp))
        z0 = d0.sample(n_samp).permute(2, 0, 1)
        d1 = Normal(3 * torch.ones(n_exp), torch.exp(theta[:, 1]/3))
        z1 = d1.sample(n_samp).permute(2, 0, 1)
        d2_1 = Normal(-2 * torch.ones(n_exp), torch.ones(n_exp))
        d2_2 = Normal(2 * torch.ones(n_exp), .5 * torch.ones(n_exp))
        d2_b = Bernoulli(.5 * torch.ones(n_exp))
        z2_b = d2_b.sample(n_samp).float()
        # Gaussian Mixture
        z2 = (z2_b * d2_1.sample(n_samp) + (1 - z2_b) * d2_2.sample(n_samp)).permute(2, 0, 1)
        d3 = Exponential(3 * torch.ones(n_exp))
        z3 = d3.sample(n_samp).permute(2, 0, 1)
        d4 = Exponential(.5 * torch.ones(n_exp))
        z4 = d4.sample(n_samp).permute(2, 0, 1)
        z = torch.cat((z0, z1, z2, z3, z4), 2)
        X = torch.matmul(self.R, z.view(-1, 5).unsqueeze(2)).view(n_exp, -1, 5)
        return X

    def get_mle(self, samples):
        n_exp = samples.shape[0]
        R_inv = torch.inverse(self.R)
        z = torch.matmul(R_inv, samples.view(-1, 5).unsqueeze(2)).view(n_exp, -1, 5)
        thetas = torch.cat((z[:, :, [0]].mean(1), z[:, :, [1]].mean(1)), 1)
        return thetas, 0*thetas
