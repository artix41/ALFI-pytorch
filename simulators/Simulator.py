import random
import torch


class Simulator:
    # The dimension of the parameter space.
    theta_dim = 0
    # The dimension of the observation space.
    x_dim = 0

    def __init__(self, rand_seed=42):
        random.seed(rand_seed)

    '''
    This method generates problems and their observations.
    IN:     n_exp:  int, the number of problems to generate.
            n_samp: int, the number of observations for each problem.
    OUT:    thetas: n_exp x theta_dim tensor, it contains the parameters of each problem.
            X:      n_exp x n_samp x x_dim, it contains the observations of each problem.
    '''
    def get_data(self, n_exp, n_samp):
        raise NotImplementedError

    '''
    This method generates the observations for specific parameters values.
    IN:     theta:  n_exp x theta_dim tensor, it contains the parameters of each problem.
            n_samp: int, the number of observations for each problem.
    OUT:    X:      n_exp x n_samp x x_dim, it contains the observations of each problem.
    '''
    def forward(self, theta, n_samp=1):
        raise NotImplementedError

    '''
    This method generates a realistic set of problems by generating parameters values.
    IN:     n_exp:  int, the number of problems to generate.
    OUT:    thetas: n_exp x theta_dim, it contains the parameters of each problem.
    '''
    def get_thetas(self, n_exp):
        raise NotImplementedError

    '''
    This method computes the maximum likelihood estimator (MLE) of the sets of observations
    IN:     samples:  n_exp x n_samp x x_dim, the sets (n_exp sets) of observations.
    OUT:    mles:     n_exp x theta_dim, it contains the MLE of the parameters of each set of observation.
            mles_std: n_exp x theta_dim, it contains the univariate standard deviation of each estimated parameter.
    '''
    def get_mle(self, samples):
        pass

    @staticmethod
    def atanh(y, epsilon=1e-20):
        return .5*(torch.log(1 + y)/(1 - y))
