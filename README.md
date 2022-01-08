# CycleGan-PyTorch

PyTorch implementation of CycleGAN. Original paper by Jun-Yan Zhu, Taesung Park, Philip Isola and Alexei A. Efros titled Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks. Link to the paper: https://arxiv.org/pdf/1703.10593.pdf 

## Acknowledgements

Some functions, ideas and architectures were taken from the following repositories:

- CycleGAN-PyTorch by Lornatang
    - https://github.com/Lornatang/CycleGAN-PyTorch
- Video Super Resolution by LoSealL
  - https://github.com/LoSealL/VideoSuperResolution

## Setup

Run the following commands to clone and setup the repository using pip

```
git clone https://github.com/ebertsoulakis/CycleGan-PyTorch.git
cd CycleGan-Pytorch
pip install -r requirements.txt
```

If using conda, run the following commands

```
conda create --file requirements.txt
conda activate cyclegan
git clone https://github.com/ebertsoulakis/CycleGan-PyTorch.git
cd CycleGan-Pytorch
```

This implementation was created using PyTorch which can be install using the following pip command
```
pip3 install torch torchvision torchaudio
```

Using conda
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Usage

Currently, only training with one dataset is possible with this repository. An example training command is shown below

```
python train.py --yml /path/to/yml --dataset /path/to/dataset --cuda --train --save_dir /path/to/save/dir
```

Hyperparameters can be changed in CycleGan.yml using an text editor.

To finetune on a pretrained checkpoint, set the 'pretrain' command as shown below

```
python train.py --yml /path/to/yml --dataset /path/to/dataset --cuda --train --save_dir /path/to/save/dir --pretrain /path/to/checkpoint/
```

## TODO

1. Add inference script
2. Add support for different datasets


