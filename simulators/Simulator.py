import random
import torch


class Simulator:
    theta_dim = 0

    def __init__(self, rand_seed=42):
        random.seed(rand_seed)

    def get_data(self, n_exp, n_samp):
        pass

    def forward(self, theta):
        pass

    def parametrize(self, theta):
        pass

    def parametrize_inv(self, theta_param):
        pass

    def get_thetas(self, n_exp):
        pass

    @staticmethod
    def atanh(y, epsilon=1e-20):
        return .5*(torch.log(1 + y)/(1 - y))
