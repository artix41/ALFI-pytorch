# Automatic Likelihood-Free Inference

Official PyTorch implementation of the paper *Recurrent Machines for Likelihood-Free Inference*, published at the NeurIPS 2018 Workshop on Meta-Learning.

## Installation

* Clone the repository:
```bash
git clone git@github.com:artix41/ALFI-pytorch.git
cd ALFI-pytorch/
```
* Inside an [Anaconda](https://www.anaconda.com/) environment, install the requirements from the file `requirements.txt`:
```bash
conda install --file requirements.txt
```

## Usage

### General

The script `run-experiments.py` starts one or several experiments described in the configuration file(s) given in input. Its general usage is `run-experiments.py [-h] [-l] [-v {0,1,2,3}] configs_dir output_dir`. For instance, if you want to run the default experiment on the simplified particle physics simulator (called Weinberg), you can run:
```batch
run-experiments.py configs/weinberg.py outputs -v 3
```
Experiments are saved automatically every `save_every` iterations (given in the [configuration file](#configuration-file-cheat-sheet)). If you want to load a previously saved experiment, simply use the `-l` loading option:
```batch
run-experiments.py configs/weinberg.py outputs -v 3 -l
```

### Use your own simulator

### Configuration file cheat sheet
Hereunder is a complete list of the configuration parameters with their description:
- model: LuckyALFINet
- simulator:
    - name:
      - ```batch
      The class name of a simulator implementation.
      e.g. PoissonSimulator, LinearRegressionSimiulator, WeinbergSimulator, ...
      ```
    - parameters: The list of parameters taken by the constructor of the simulator class. e.g. [], "sigma: 0.5", ...
- train:
    - lr_scheduling: false
    - nb_iter_alfi_min: 15
    - nb_iter_alfi_max: 15
    - nb_iter_alfi_step: 5
    - nb_iter_alfi_period: 5
    - nb_theta: 1000
    - nb_x_per_theta: 500
    - meta_batch_size: 16
    - batch_size_x: 64
    - batch_size_theta: 8
    - lr: 0.0002
    - save_every: 5
    - test_every: 5
    - loss: normal
    - weight: exp
    - nb_epochs: 130
- test:
    - nb_iter_alfi: 15
    - nb_theta: 1000
    - nb_x_per_theta: 500
    - meta_batch_size: 16
    - batch_size_x: 64
    - batch_size_theta: 8
- networks:
    - use_grad: true
    - split_theta: false
    - x_data_agg:
        - hidden_size: 50
        - output_size: 60
    - theta_data_agg:
        - hidden_size: 50
        - output_size: 60
    - RIM:
        - hidden_size: 50
        - st_size: 40
        - bounded: 0.2
- proposal:
    - name: GaussianProposal
    - sigma: 0.1
- plot:
    - title: True
    - rmse_t:
        - nb_theta: 12
    - hist-x:
        - nb_theta: 6
        - nb_samples_real: 5000
        - nb_samples_gen: 5000
        - nb_bins: 10
    - init_box:
        - nb_theta: 12
    - init_rmse_t:
        - nb_theta: 12
    - init_comparison:
        - nb_inits: 20
        - nb_theta: 500
    - init_MLE:
        - nb_theta: 12




## Contacts

If you have any question, please contact [Antoine Wehenkel](https://github.com/AWehenkel) (antoine.wehenkel@uliege.be) or [Arthur Pesah](https://artix41.github.io) (arthur.pesah@gmail.com).
