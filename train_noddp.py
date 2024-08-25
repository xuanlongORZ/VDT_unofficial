# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader

from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from dataset import VideoDataset
from mask_generator import VideoMaskGenerator


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    device = 'cuda:0'
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = VDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="", training=True)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = VideoDataset(args.data_path, frames_per_clip=args.num_frames, transform=transform, mask_ratio=0.8)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(total=len(loader), desc=f'Epoch {epoch}/{args.epochs}', unit='sample') as pbar:
            for idx, clip in enumerate(loader):
                clip = clip.to(device)
                B, T, C, H, W = clip.shape
                # x = x.view(-1, C, H, W).to(device=device)
                clip = clip.view(-1, C, H, W).to(device=device)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(clip).latent_dist.sample().mul_(0.18215)
                generator = VideoMaskGenerator((T,H//8,W//8)) # t,h,w
                mask = generator(batch_size=4, idx=2).unsqueeze(0) # 1,C,T,H,W
                mask = 1 - mask.permute(0,2,1,3,4).to(device=clip.device) # 1,T,C,H,W
                x = x.view(-1, args.num_frames, 4, x.shape[-2], x.shape[-1]) # B,T,C,H,W
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                # model_kwargs = dict(y=y)
                # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                # print(mask.shape, x.shape, t.shape)
                loss_dict = diffusion.training_losses(model = model, x_start = x, t = t, mask = mask)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                update_ema(ema, model)


                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1

                # update bar with latest loss
                pbar.set_postfix(loss=running_loss/(idx+1))
                pbar.update(1)

                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss = avg_loss.item()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "pos_embed": model.pos_embed,  # 将位置嵌入存入checkpoint中
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...a

    logger.info("Done!")
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--image-size", type=int,  default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=2)

    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--data-path", type=str, default='/root/autodl-tmp/VDT_unofficial/datasets/UCF-101')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=6000)
    args = parser.parse_args()
    main(args)