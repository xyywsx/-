# 
步态识别是一种特别的生物识别技术。针对步态识别中特征提取的局限性和对浅层特征的关注度不足，提出了一种基于多分支特征融合的步态识别网络

# Get Started
## Installation
1. clone this repo.
    ```
    git clone https://github.com/xyywsx/Research-on-gait-recognition-network-based-on-multi-branch-feature-fusion.git
    ```
2. Install dependenices:
    - pytorch >= 1.10
    - torchvision
    - pyyaml
    - tensorboard
    - opencv-python
    - tqdm
    - py7zr
    - kornia
    - einops
  
    Install dependenices by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
    ```
    conda install tqdm pyyaml tensorboard opencv kornia einops -c conda-forge
    conda install pytorch==1.10 torchvision -c pytorch
    ```    
    Or, Install dependenices by pip:
    ```
    pip install tqdm pyyaml tensorboard opencv-python kornia einops
    pip install torch==1.10 torchvision==0.11
    ```
## Prepare dataset
See [prepare dataset](prepare_dataset.md).

## Train
Train a model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitmbgl/gaitmbgl.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) for training different models.

## Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitmbgl/gaitmbgl.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify a iteration checkpoint.

**Tip**: Other arguments are the same as train phase.

You can run commands in [test.sh](test.sh) for testing different models.

## Customize
1. Read the [detailed config](docs/1.detailed_config.md) to figure out the usage of needed setting items;
