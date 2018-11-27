import torch

class ConstantProposal():
    psi_dim = 1
    is_random = False

    def __init__(self, psi=None):
        super().__init__()
        self.psi = psi  # Tensor of size [nb_psi, psi_dim, theta_dim]

    def sample(self, sample_size):
        samples = self.psi.expand(-1, sample_size, -1)
        return samples

    def update_psi(self, psi):
        self.psi = psi

    def init_psi(self, nb_psi, simulator,mu_theta=0, sigma_theta=1, device=torch.device("cpu")):
        self.psi = (simulator.get_thetas(nb_psi).view(nb_psi, 1, -1).to(device) - mu_theta)/sigma_theta
        return self.psi
