from simulators import Simulator

import torch
from torch.distributions import Uniform
from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)
    
class WeinbergSimulator(Simulator):
    theta_dim = 2
    x_dim = 1
    
    def __init__(self, rand_seed=42, device="cpu"):
        super().__init__(rand_seed)

        self.min_theta = torch.FloatTensor([40., 0.5])
        self.max_theta = torch.FloatTensor([50., 1.5])
    
    def a_fb(self, sqrtshalf, gf):
        MZ = 90.
        GFNom = 1.
        sqrts = sqrtshalf * 2.
        A_FB_EN = torch.tanh((sqrts - MZ) / MZ * 10)
        A_FB_GF = gf / GFNom

        return 2 * A_FB_EN*A_FB_GF

    def diffxsec(self, costheta, sqrtshalf, gf):
        norm = 2. * (1. + 1. / 3.)
        return ((1 + costheta ** 2) + self.a_fb(sqrtshalf, gf).reshape(-1,1) * costheta) / norm

    def rej_sample_costheta(self, n_samp, theta, rej_matrix_size):
        n_theta = theta.shape[0]
        sqrtshalf = theta[:,0]
        gf = theta[:,1]
        
        x = torch.linspace(-1, 1, 2000).repeat(n_theta, 1)
        maxval = torch.max(self.diffxsec(x, sqrtshalf, gf), dim=1)[0].reshape(-1,1)

        BIG_NUMBER = 2. # number bigger than all the samples
        
        X_real = torch.zeros((n_theta, n_samp))
        ntrials = 0
        current_idx_sample = 0
        nb_samples_left = n_samp
        n_samp_new = min(100,n_samp*50)

        while nb_samples_left > 0:
            ntrials += 1
            # print("trial : {}, nb_samples_left: {}".format(ntrials, nb_samples_left))
            
            X_new = Uniform(-1, 1).sample(torch.Size([n_theta, n_samp_new]))

            ycut = torch.rand(n_theta, n_samp_new)
            yprop = self.diffxsec(X_new, sqrtshalf, gf)

            X_new[(yprop / maxval) < ycut] = BIG_NUMBER
            idx_new = (X_new==BIG_NUMBER).sort(dim=1)[1]
            X_new = batched_index_select(X_new, dim=1, index=idx_new)

            nb_max_big_numbers = int((X_new == BIG_NUMBER).sum(dim=1).max())
            
            if n_samp_new - nb_max_big_numbers != 0:
                X_new = X_new[:, :n_samp_new - nb_max_big_numbers]
                size_new = n_samp_new - nb_max_big_numbers

                size_for_complete = min([size_new, nb_samples_left, n_samp - current_idx_sample])
                X_real[:,current_idx_sample:current_idx_sample + size_for_complete] = X_new[:,:size_for_complete]

                nb_samples_left -= size_for_complete
                current_idx_sample = n_samp - nb_samples_left 

        # print("nb trials : ", ntrials)
        return X_real.reshape(n_theta,n_samp,1)

    def forward(self, theta, n_samp=1, rej_matrix_size=1000):
        samples = self.rej_sample_costheta(n_samp, theta, rej_matrix_size)

        return samples
        
    def get_thetas(self, n_exp):
        thetas = Uniform(self.min_theta, self.max_theta).sample(torch.Size([n_exp]))

        return thetas
        
    def get_data(self, n_exp, n_samp, rej_matrix_size=1000):
        thetas = self.get_thetas(n_exp)
        X_real = self.forward(thetas, n_samp, rej_matrix_size)
        
        return thetas, X_real
