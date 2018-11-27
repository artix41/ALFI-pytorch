import torch
from torch.distributions import Normal

class FixedVarianceGaussianProposal():
    psi_dim = 1
    is_random = True
    def __init__(self, config, psi=None):
        super().__init__()
        self.psi = psi
        self.config = config
        self.sigma = config["sigma"]
        
    def sample(self, sample_size):
        samples = Normal(self.psi[:, 0, :], self.sigma).sample(torch.Size([sample_size]))
        samples = samples.permute(1, 0, 2)
        return samples

    def update_psi(self, psi):
        self.psi = psi

    def init_psi(self, nb_psi, simulator, mu_theta=0, sigma_theta=1, device=torch.device("cpu")):
        self.psi = (simulator.get_thetas(nb_psi).view(nb_psi, 1, -1).to(device) - mu_theta)/sigma_theta
        
        return self.psi

    def grad_log(self, thetas):
        psi = self.psi.clone()
        grad_mu = (psi[:, 0, :] - thetas)/(self.sigma**2)
        grad_psi = grad_mu.unsqueeze(1)
        
        return grad_psi
