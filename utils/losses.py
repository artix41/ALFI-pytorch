import torch


def loss_mse(theta, list_psi_t):
    return ((list_psi_t[:, 0] - theta)**2).mean(dim=1)


def loss_l1(theta, list_psi_t):
    return torch.abs(list_psi_t[:, 0] - theta).mean(dim=1)


def loss_normal_density(theta, list_psi_t):
    return -torch.distributions.Normal(list_psi_t[:, 0], torch.exp(list_psi_t[:, 1])).log_prob(theta).sum(1)
