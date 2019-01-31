from models import *
import os
from simulators import PoissonSimulator
import yaml
import sys
from collections import Mapping
from simulators import *
from proposals import *

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config


def load_trained_model(dir, device="cpu", verbose=3):
    # Load the config file
    config = load_config(os.path.join(dir, os.path.basename(os.path.normpath(dir)) + '.yaml'))

    # Split the configs part
    sim_config = config["simulator"]
    prop_config = config["proposal"]
    net_config = config["networks"]
    train_config = config["train"]
    test_config = config["test"]
    plot_config = config["plot"]

    Model = str_to_class(config["model"])

    if isinstance(sim_config["parameters"], Mapping):
        simulator = str_to_class(sim_config["name"])(**sim_config["parameters"], device=device)
    else:
        simulator = str_to_class(sim_config["name"])(*sim_config["parameters"], device=device)
    proposal = str_to_class(prop_config["name"])(prop_config, device=device)

    alfi = Model(net_config, train_config, simulator, proposal, verbose=verbose, device=device)
    if verbose >= 1:
        print("[*] Loading existing model...")
    alfi.load(os.path.join(dir, 'model'), os.path.join(dir, 'values'))

    return alfi

def usage_example():
    alfi = load_trained_model("output/poisson/")
    sim = PoissonSimulator()

    nb_X = 100
    nb_theta = 3

    theta, X = sim.get_data(nb_theta, nb_X)

    # theta_est.shape : [nb_theta, nb_iter, psi_dim, theta_dim]
    theta_est = alfi.forward(X, batch_size_x=nb_X, batch_size_theta=16, nb_iter=10, phase="test").detach().cpu()

    # Print the value of the first proposal parameter reached at the end of the iterative procedure.
    print(theta_est[:, -1, 0, :], theta)