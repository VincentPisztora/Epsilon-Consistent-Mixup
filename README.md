# Epsilon-Consistent-Mixup

<div align="center">
  <img src="/assets/lambda_eta_schedule.png" alt="image" width="400">
  <br>
  <small><em>This figure illustrates how the ϵmu response mixing parameter ηϵ(λ) relates to the feature mixing parameter λ across several values for the rescaled consistency neighborhood radius $`v_{\epsilon}`$</em></small>
</div>
<br>

This repository contains the code referenced in the Stat paper "Epsilon Consistent Mixup: An Adaptive Consistency-Interpolation Tradeoff" ([arxiv](https://arxiv.org/abs/2104.09452)) ([Stat](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.425)).

In this paper we propose Epsilon Consistent Mixup ($`\epsilon`$mu). $`\epsilon`$mu is a data-based structural regularization technique that combines Mixup's linear interpolation with consistency regularization in the Mixup direction, by compelling a simple adaptive tradeoff between the two. This learnable combination of consistency and interpolation induces a more flexible structure on the evolution of the response across the feature space and is shown to improve semi-supervised classification accuracy on the SVHN and CIFAR10 benchmark datasets, yielding the largest gains in the most challenging low label-availability scenarios. Empirical studies comparing $`\epsilon`$mu and Mixup are presented and provide insight into the mechanisms behind $`\epsilon`$mu's effectiveness. In particular, $`\epsilon`$mu is found to produce more accurate synthetic labels and more confident predictions than Mixup.

Please find below a step-by-step guide for training the models described in the paper.

Setup:

0. Create the python environment
   - The packages needed to run all code are specified in the env_emu.yml file.
   - The environment can be built using this yml file as described here: [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

1. Set directories
   - data.py: Set `DATA_DIR` variable
   - create_dataset.py: Set `PROJECT_DIR` variable
   - emu.py: Set `data_dir` and `models_dir` variables

2. Download datasets
   - Run: `CUDA_VISIBLE_DEVICES=0 python /project/directory/create_datasets.py`

3. Training
   - Run: `CUDA_VISIBLE_DEVICES=0 python emu.py data seed n_label n_valid aug method arch wd w_u beta eps lr ema`
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

<img src="https://www.google-analytics.com/collect?v=1&tid=G-QST3V3PB55&cid=555&t=event&ec=repo&ea=view&el=Epsilon-Consistent-Mixup" style="display:none">
