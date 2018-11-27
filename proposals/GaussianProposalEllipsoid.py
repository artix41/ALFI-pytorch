import torch
from torch.distributions import Normal

class GaussianProposalEllipsoid():
    psi_dim = 2
    is_random = True
    def __init__(self, config, psi=None):
        super().__init__()
        self.psi = psi
        self.config = config

    def sample(self, sample_size):
        sigmas = torch.exp(self.psi[:, 1, :])
        sigmas_max = 3*sigmas
        sigmas_min = .05*sigmas
        #print(sigmas_min)
        step_sigmas = (sigmas_max - sigmas_min)/(sample_size - 1)
        samples = torch.zeros(self.psi.shape[0], sample_size, self.psi.shape[2])
        for i in range(sample_size):
            el_axes = self.create_diag(sigmas_min + i*step_sigmas)

            sphere_contour = Normal(loc=0, scale=1).sample(self.psi[:, 1, :].size())
            if (torch.norm(sphere_contour, p=2, dim=1) < 1e-5).sum() > 0:
                print(torch.norm(sphere_contour, p=2, dim=1))
            unit_sph_cont = (sphere_contour/torch.norm(sphere_contour, p=2, dim=1).unsqueeze(1)).unsqueeze(2)
            mu = self.psi[:, 0, :]
            samples[:, i, :] = torch.matmul(el_axes, unit_sph_cont)[:, :, 0] + mu
        return samples

    def update_psi(self, psi):
        self.psi = psi

    def init_psi(self, nb_psi, simulator, mu_theta=0, sigma_theta=1, device=torch.device("cpu")):
        mu = (simulator.get_thetas(nb_psi).view(nb_psi, 1, -1).to(device) - mu_theta)/sigma_theta
        sigma = torch.log(self.config["sigma"] * torch.ones(mu.size())).to(device)
        self.psi = torch.cat((mu, sigma), 1)
        
        return self.psi

    def grad_log(self, thetas):
        psi = self.psi.clone()
        sigma = torch.exp(psi[:, 1, :])
        grad_mu = (psi[:, 0, :] - thetas)/(sigma**2)
        grad_sigma = 0*(((psi[:, 0, :] - thetas)**2)/sigma**2 - 1)
        grad_psi = torch.cat((grad_mu.unsqueeze(1), (sigma*grad_sigma).unsqueeze(1)), 1)
        return grad_psi

    '''
    Takes a: [d1 x d2] and returns [d1 x d2 x d2] tensor where the two last dimensions make diagonal matrices
    '''
    def create_diag(self, a):
        b = torch.eye(a.size(1))
        c = a.unsqueeze(2).expand(*a.size(), a.size(1))
        return c * b
