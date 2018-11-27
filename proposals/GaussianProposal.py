import torch
from torch.distributions import Normal

class GaussianProposal():
    psi_dim = 2
    is_random = True
    def __init__(self, config, psi=None, device="cpu"):
        super().__init__()
        self.psi = psi
        self.config = config
        self.device = device

    def sample(self, sample_size):
        sigma = torch.exp(self.psi[:, 1, :])
        samples = Normal(self.psi[:, 0, :], sigma).sample(torch.Size([sample_size]))
        samples = samples.permute(1, 0, 2)
        return samples

    def update_psi(self, psi):
        self.psi = psi

    def init_psi(self, nb_psi, simulator, mu_theta=0, sigma_theta=1):
        mu = (simulator.get_thetas(nb_psi).view(nb_psi, 1, -1) - mu_theta)/sigma_theta
        sigma = torch.log(self.config["sigma"] * torch.ones(mu.size())).to(self.device)
        self.psi = torch.cat((mu, sigma), 1)
        return self.psi

    def grad_log(self, thetas):
        psi = self.psi.clone()
        sigma = torch.exp(psi[:, 1, :])
        grad_mu = (psi[:, 0, :] - thetas)/(sigma**2)
        grad_sigma = (((psi[:, 0, :] - thetas)**2)/sigma**2 - 1)
        grad_psi = torch.cat((grad_mu.unsqueeze(1), (sigma*grad_sigma).unsqueeze(1)), 1)
        return grad_psi

    def normalized_grad_log(self, thetas):
        grad = self.grad_log(thetas)
        mu = grad.mean(0)
        sigma = grad.std(0)
        return (grad - mu)/sigma
