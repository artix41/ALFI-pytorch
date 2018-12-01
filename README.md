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
    - `It contains the parameters related to the testing configuration.`
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
    - `It contains the parameters related to the training configuration.`
    - use_grad:
      - `Put the variational gradient (`<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_{\psi}q(\theta|\psi)" title="\Large \nabla_{\psi}q(\theta|\psi)" />`) as input or not.`
      - `e.g. true, false.`
    - split_theta:
      - `Whether consider the optimization over each parameter independently or not.`
      - `e.g. true, false.`
    - x_data_agg:
        - `Configuration parameters of the network that encodes the observations.`
        - hidden_size:
          - `The number of units by hidden layer.`
          - `e.g. 50.`
        - output_size: 60
          - `The number of output units.`
          - `e.g. 50.`
    - theta_data_agg:
      - `Configuration parameters of the network that aggregates the encoded observations and the gradients.`
      - hidden_size:
        - `The number of units by hidden layer.`
        - `e.g. 50.`
      - output_size: 60
        - `The number of output units.`
        - `e.g. 50.`
    - RIM:
        - `Configuration parameters of the RNN.`
        - hidden_size:
          - `The number of units by hidden layer.`
          - `e.g. 50.`
        - st_size:
        - `The number of memory units of the GRU.`
        - `e.g. 50.`
        - bounded:
          - `The bound on the output of the RNN which represents the update step on the proposal distribution (a negative number if no bound). `
          - `e.g. -1, 0.2`
- proposal:
    - name:
      - `The class name of the proposal distribution on parameters value.`
      - `e.g. ConstantProposal, FixedVarianceGaussianProposal, GaussianProposal`
    - sigma:
        - `A parameter value of the constructor of the proposal, it could be something else than sigma.`
        - `e.g. 0.5.`
- plot:
    - title:
      - `Whether or not the figures should contains a title.`
      - `e.g. true, false.`
    - rmse_t:
        - nb_theta:
          - `The number of theta to produce the plot that shows the evolution of the RMSE along iterations for these parameters.`
          - `e.g. 12.`
    - hist-x:
        - nb_theta:
          - `The number of theta to produce the plot that compares the observations generated with the final proposal distribution output by ALFI with the observations of the testing set.`
          - `e.g. 6.`
        - nb_samples_real:
          - `The number of observations used to do the plot for the true parameters.`
          - `e.g. 5000.`
        - nb_samples_gen:
          - `The number of observations used to do the plot for the estimated parameters.`
          - `e.g. 5000.`
        - nb_bins:
          - `The number of bins in the histograms.`
          - `e.g. 10.`
    - init_box:
        - `Box plot that shows what is the variation of the final proposal (after T iterations) depending on the starting proposal.`
        - nb_theta:
          - `The number of theta and so the number of box plot.`
          - `e.g. 12`
    - init_rmse_t:
        - `This plot shows what is the impact of different starting proposal on the evolution of the RMSE along iterations.`
        - nb_theta:
          - `The number of theta and so the number of sub plots.`
          - `e.g. 12`
    - init_comparison:
        - `Configuration to pre-compute data for the two plots above.`
        - nb_inits:
          - `The number of starting proposal.`
          - `e.g. 12.`
        - nb_theta:
          - `The number of theta (should be greater or equal than for the two previous plots configurations.)`


## Contacts

If you have any question, please contact [Antoine Wehenkel](https://github.com/AWehenkel) (antoine.wehenkel@uliege.be) or [Arthur Pesah](https://artix41.github.io) (arthur.pesah@gmail.com).
