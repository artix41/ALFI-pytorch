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
Experiments are saved automatically every `save_every` iterations (given in the [configuration file](configs/README.md#configuration-file-cheat-sheet)). If you want to load a previously saved experiment, simply use the `-l` loading option:
```batch
run-experiments.py configs/weinberg.py outputs -v 3 -l
```

### Use your own simulator
To implement your own simulator you should write a class that inherits from [Simulator.py](simulators/Simulator.py).


## Contacts

If you have any question, please contact [Antoine Wehenkel](https://github.com/AWehenkel) (antoine.wehenkel@uliege.be) or [Arthur Pesah](https://artix41.github.io) (arthur.pesah@gmail.com).
