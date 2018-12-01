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
- model:
  - `The class name of the model to train. You can for example change the code of ALFINet.py into a file BetterALFINet.py`
  - ` e.g. LuckyALFINet, BetterALFINet`
- simulator:
    - `It contains the parameters related to the simulator.`
    - name:
      - `The class name of a simulator implementation.`
      - `e.g. PoissonSimulator, LinearRegressionSimiulator, WeinbergSimulator, ...`
    - parameters:
      - `The list of parameters taken by the constructor of the simulator class.`
      - ` e.g. [], sigma: 0.5, ...`
- train:
    - `It contains the parameters related to the training configuration.`
    - lr_scheduling:
      - `True to use sheduling on the optimizer during training, the sheduling parameters are predefined in the code of the model.`
      - `e.g. true, false`
    - nb_iter:
      - `The number of iteration of ALFI (T in the manuscript).`
      - `e.g. 15`
    - nb_theta:
      - `The number of thetas (simulation parameters) in the meta training set. Each of them constitute a problem.`
      - `e.g. 1000`
    - nb_x_per_theta:
      - `The number of observations for each theta in the meta training set.`
      - `e.g. 500`
    - meta_batch_size:
      - `The number of problem taken from the training to perform one gradient update.`
      - `e.g. 16`
    - batch_size_theta:
      - `The number of thetas drawn from the random distribution proposal on the true simulation parameters.`
      - `e.g. 8`
    - batch_size_x:
      - `The number of observations generated for each of these theta.`
      - `e.g. 64`
    - lr:
      - `The learning rate.`
      - `e.g. 0.0002`
    - save_every:
      - `The number of epochs between two saveguard.`
      - `e.g. 5`
    - test_every:
      - `The number of epochs between two tests (useful to do training plots).`
      - `e.g. 5`
    - loss:
      - `The type of loss used. If you want to write your own it must be relevant to compare the proposal distribution with the true parameters value.`
      - `e.g. normal, MSE, l1`
    - weight:
      - `The weighting on the loss which should controls the tradeoff exploration/exploitation of the iterative process.`
      - `e.g. exp, oi, last, constant`
    - nb_epochs:
      - `The number of training epochs.`
      - `e.g. 130`
- test:
    - nb_iter_alfi:
      - `Same meaning as for train (but the value can be different).`
    - nb_theta:
      - `The number of different simulation parameters for test.`
      - `e.g. 1000`
    - nb_x_per_theta:
    - `Same meaning as for train (but the value can be different).`
    - meta_batch_size:
      - `Same meaning as for train (but the value can be different). It is only useful to limit memory requirements.`
    - batch_size_x:
      - `Same as for train.`
    - batch_size_theta:
      - `Same as for train.`
- networks:
    - use_grad:
      - `Put the variational gradient (<html>&Nabla;<sub>&psi;</sub>q(&theta;|&psi;)</html>) as input or not.`
      - `e.g. true, false`
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
