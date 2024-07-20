# VDT - unofficial

This repo is to implement the training part for VDT (VDT: General-purpose Video Diffusion Transformers via Mask Modeling [ICLR2024]) for my own interest. I am more interested in the frame interpolation performance of this framework, hoping it can outperform the results of the 'crafter series' (Tooncrafter/DynamiCrafter). If I violate any open source agreement/licences, please inform me in time, thank you.

<img src="VDT.png" width="700">

## Introduction and logs

1. I just adjust and apply DiT training script to VDT: train_noddp.py and train.py. train_noddp.py is the 'no Distributed Data Parallelism' version of train.py, but since I only have one card, I didn't run train.py but only train_noddp.py on my side.
2. Todo:
   1. Evaluation part
   2. More careful mask design to reproduce the training in the article

## Getting Started

- Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.
- To install the other dependencies, run
  `<pre/>`conda env create -f environment.yml `</pre>`
  `<pre/>`conda activate VDT `</pre>`

## Checkpoint

The author now provide checkpoint for Sky Time-Lapse unified generation. You can download it from  `<a href="https://drive.google.com/file/d/1WIAOm4n0HkmOHMhUj3y6wLLemtz_Xj8b/view?usp=sharing">` here `</a>`.

## Train

Run python train_noddp.py. The arguments should be adjusted for example data-path. For the moment, only train_noddp.py is ok to run on my side.

## Inference

The authors provide inference ipynb on Sky Time-Lapse unified generation (predict, backward, unconditional, single-frame, arbitrary interpolation, spatial_temporal). To sample results, you can first download the checkpoint, then run inference.ipynb, have fun! But you'd better try the inference.py version in this repo because I adjust a bit the files in the diffusion folder (mostly about the dimension changing part).

## Acknowledgement

The original codebase is built based on DiT, BEiT, SlotFormer and MVCD. And the training part is also heavily based on DiT. We thank the authors for the nicely organized code!
