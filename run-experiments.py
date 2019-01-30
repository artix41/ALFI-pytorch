import numpy as np
import yaml
import argparse
import glob
import os
import shutil
from experiment import experiment

# ================== Arguments ==================
    
parser = argparse.ArgumentParser(
    description='Run all experiments described in the different config files'
)
parser.add_argument('configs_dir', type=str, help='Path to configs directory or to a config file')
parser.add_argument('output_dir', type=str, help='Path to the output directory')
parser.add_argument('-l', '--load', action="store_true", help='Load the last model for already existing experiments')
parser.add_argument('-v', choices=['0','1','2','3'], default=3, dest='verbose',
                    help='Level of verbosity, between 0 and 3')
args = parser.parse_args()

output_dir = os.path.abspath(args.output_dir)
configs_dir = os.path.abspath(args.configs_dir)
verbose = args.verbose
load = args.load

# ================ Load config files =============

if os.path.isdir(configs_dir):
    config_files = glob.glob(os.path.join(configs_dir, '*.yaml'))
else:
    config_files = [configs_dir]

experiment_names = [os.path.splitext(os.path.basename(f))[0] for f in config_files]
    
def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config
    
for i_exp in range(len(config_files)):
    exp_name = experiment_names[i_exp]
    verbose = int(verbose)
    if verbose >= 1:
        print("\n===========================================================")
        print("============== Running Experiment {:02d}: {} ==============".format(i_exp, exp_name))
        print("===========================================================\n")
        print("[*] Loading config files and creating output directories...\n")

    config = load_config(config_files[i_exp])
    exp_output_dir = os.path.join(output_dir, exp_name)

    if not os.path.exists(exp_output_dir) or (os.path.exists(exp_output_dir) and not load):
        if os.path.exists(exp_output_dir) and not load:
            shutil.rmtree(exp_output_dir)
        os.makedirs(exp_output_dir)
        shutil.copy(config_files[i_exp], exp_output_dir)
    
    
    if verbose >= 1:
        print("[*] Starting experiment...\n")
        
    experiment(config, exp_output_dir, load=load, verbose=verbose)

if verbose >= 1:
    print("\n=================== Done ===================\n")
