# Epsilon-Consistent-Mixup

This repository contains the code referenced in the Stat paper "Epsilon Consistent Mixup: An Adaptive Consistency-Interpolation Tradeoff" (https://arxiv.org/abs/2104.09452).

Please find below a step-by-step guide for training the models described in the paper.

Setup:
0a. Create the python environment
   - The packages needed to run all code are specified in the "env_emu.yml" file.
   - The environment can be built using this yml file as described here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
0b. Set directories
   - "data.py": Set "DATA_DIR" variable
   - "create_dataset.py": Set "PROJECT_DIR" variable
   - "emu.py": Set "data_dir" and "models_dir" variables
1. Download datasets
   - Run: CUDA_VISIBLE_DEVICES=0 python /project/directory/create_datasets.py

Training:
2. Run: CUDA_VISIBLE_DEVICES=0 python emu.py data seed n_label n_valid aug method arch wd w_u beta eps lr ema
   - For each:
     - data: 'cifar10', 'svhn'
     - seed: 0, 1, 2
     - n_label: 40, 250
     - n_valid: 1000
     - aug: 'y', 'n'
     - method: 'mu', 'emu'
     - arch: 'WideResNet28_2'
     - wd: 
       - cifar10: 0.06, 0.12, 
       - svhn: 0.18, 0.20, 0.30, 0.50, 0.60, 0.90
     - w_u: 1, 10, 20, 50, 100
     - beta: 0.1, 0.2, 0.5, 1.0
     - eps: 0.0, 10.0
     - lr: 0.002
     - ema: 0.999
