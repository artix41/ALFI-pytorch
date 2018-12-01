import random
import torch


class Simulator:
    theta_dim = 0

    def __init__(self, rand_seed=42):
        random.seed(rand_seed)

    def get_data(self, n_exp, n_samp):
        raise NotImplementedError

    def forward(self, theta):
        raise NotImplementedError

    def get_thetas(self, n_exp):
        raise NotImplementedError

    def get_mle(self, samples):
        pass

    @staticmethod
    def atanh(y, epsilon=1e-20):
        return .5*(torch.log(1 + y)/(1 - y))
