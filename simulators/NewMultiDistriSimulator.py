from simulators import Simulator

import torch
from torch.distributions import Uniform, Normal, Bernoulli, Exponential

class NewMultiDistriSimulator(Simulator):
    theta_dim = 3
    x_dim = 5
    R = torch.tensor([[ 1.31229955,  0.10499961,  0.48310515, -0.3249938,  -0.26387927],
                     [ 0.10499961,  1.15833058, -0.55865473,  0.25275522, -0.39790775],
                     [ 0.48310515, -0.55865473,  2.25874579, -0.52087938, -0.39271231],
                     [-0.3249938,   0.25275522, -0.52087938,  1.4034925,  -0.63521059],
                     [-0.26387927, -0.39790775, -0.39271231, -0.63521059,  1.        ]])

    def __init__(self, rand_seed=42, min=-3., max=3., device="cpu"):
        super().__init__(rand_seed)
        self.min = torch.tensor(min, device=device)
        self.max = torch.tensor(max, device=device)
        self.R = self.R.to(device)
        self.device = device

    def get_data(self, n_exp, n_samp):
        d = Uniform(self.min, self.max)
        thetas = d.sample(torch.Size([n_exp, self.theta_dim]))

        X = self.forward(thetas, n_samp=n_samp)

        return thetas.reshape([-1, self.theta_dim]), X

    def get_thetas(self, n_exp):
        d = Uniform(self.min, self.max)
        return d.sample(torch.Size([n_exp, self.theta_dim]))

    def forward(self, theta, n_samp=1):
        n_exp = theta.shape[0]
        n_samp = torch.Size([n_samp, 1])
        unit = torch.ones(n_exp).to(self.device)
        with torch.autograd.no_grad():
            d0 = Normal(theta[:, 0], unit)
            z0 = d0.sample(n_samp).permute(2, 0, 1)
            d1 = Normal(3 * unit, torch.exp(theta[:, 1]/3))
            z1 = d1.sample(n_samp).permute(2, 0, 1)
            d2_1 = Normal(-2 * unit, unit)
            d2_2 = Normal(2 * unit, .5 * unit)
            d2_b = Bernoulli(.5 * unit)
            z2_b = d2_b.sample(n_samp).float()
            # Gaussian Mixture
            z2 = ((z2_b * d2_1.sample(n_samp) + (1 - z2_b) * d2_2.sample(n_samp)).permute(2, 0, 1))
            d3 = Uniform(-5*unit, theta[:, 2])
            z3 = d3.sample(n_samp).permute(2, 0, 1)
            d4 = Exponential(.5 * unit)
            z4 = d4.sample(n_samp).permute(2, 0, 1)
            z = torch.cat((z0, z1, z2, z3, z4), 2)
            X = torch.matmul(self.R, z.view(-1, 5).unsqueeze(2)).view(n_exp, -1, 5)
        return X

    """
    samples: mb_size x nb_x x x_dim
    theta:   mb_size x b_size x theta_dim
    return:  mb_size x b_size
    """
    def get_likelihood(self, samples, theta):
        mb_size, nb_x, x_dim = samples.shape
        b_size = theta.shape[1]

        R_inv = torch.inverse(self.R)
        unit = torch.ones(mb_size*b_size, 1).to(self.device)
        z = torch.matmul(R_inv, samples.view(-1, 5).unsqueeze(2)).view(mb_size, nb_x, -1).unsqueeze(1)\
            .expand((mb_size, b_size, nb_x, x_dim))

        d0 = Normal(theta[:, :, 0].view(-1, 1), unit)
        log_lik_z0 = d0.log_prob(z[:, :, :, 0].view(-1, nb_x)).view(mb_size, b_size, -1)

        d1 = Normal(3 * unit, torch.exp(theta[:, :, 1].view(-1, 1)/3))
        log_lik_z1 = d1.log_prob(z[:, :, :, 1].view(-1, nb_x)).view(mb_size, b_size, -1)

        d3 = Uniform(-5*unit, theta[:, :, 2].view(-1, 1))
        log_lik_z3 = d3.log_prob(z[:, :, :, 3].view(-1, nb_x)).view(mb_size, b_size, -1)

        log_lik_tot = log_lik_z0 + log_lik_z1 + log_lik_z3
        norm_log_lik = log_lik_tot/log_lik_tot.sum(1).unsqueeze(1).expand(-1, log_lik_tot.shape[1], -1)
        return norm_log_lik.sum(2).sum(1)

    def get_likelihood_grad(self, samples, theta, psi):
        psi = psi.copy()
        psi.requires_grad = True
        d = Normal

    def get_mle(self, samples):
        n_exp = samples.shape[0]
        R_inv = torch.inverse(self.R.cpu())
        samples = samples.cpu()

        z = torch.matmul(R_inv, samples.view(-1, 5).unsqueeze(2)).view(n_exp, -1, 5)
        thetas = torch.cat((z[:, :, [0]].mean(1), torch.log(z[:, :, [1]].std(1))*3, z[:, :, [3]].max(1)[0]), 1)
        return thetas, 0*thetas
