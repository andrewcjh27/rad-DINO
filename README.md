# RAD-DINO

Fine-tuning Microsoft's RAD-DINO (a radiology-focused DINOv2 vision transformer) for medical image classification and segmentation.

## Overview

This project adapts the pretrained [RAD-DINO](https://huggingface.co/microsoft/rad-dino) model for downstream medical imaging tasks. It supports LoRA-based parameter-efficient fine-tuning, optional UPerNet segmentation decoding, and GradCAM visualization for model interpretability.

## Features

- **LoRA Fine-Tuning** — Parameter-efficient adaptation of the RAD-DINO vision transformer encoder with configurable rank and alpha
- **UPerNet Segmentation** — Optional segmentation decoder head for pixel-level predictions
- **Distributed Training** — Multi-GPU training with PyTorch DistributedDataParallel (DDP)
- **GradCAM Visualization** — Attention-based heatmap generation for model interpretability
- **Custom Data Pipeline** — Configurable dataset loaders with balanced batch sampling
- **Bias Analysis** — Training scripts for evaluating and mitigating model bias

## Tech Stack

- PyTorch
- Hugging Face Transformers (`microsoft/rad-dino`)
- Segmentation Models PyTorch (UPerNet decoder)
- torchvision, NumPy, Pillow

## Project Structure

- `rdino_model.py` — RAD-DINO model with LoRA and optional UPerNet segmentation head
- `rdino_train.py` — Single-GPU training loop
- `rdino_train_ddp.py` — Distributed Data Parallel training
- `rdino_losses.py` — Loss functions (classification + segmentation)
- `rdino_dataset.py` — Dataset and data loading utilities
- `rdino_gradcam.py` — GradCAM visualization
- `rdino_config.py` — Hyperparameter configuration
- `rdino_utils.py` — Training utilities and metrics
- `bias_train.py` — Bias evaluation training
- `custom_batch_sampler.py` — Balanced batch sampling strategy
- `custom_scheduler.py` — Learning rate scheduler

## Usage

```bash
pip install torch transformers segmentation-models-pytorch pillow numpy

# Single GPU training
python rdino_train.py

# Multi-GPU distributed training
torchrun --nproc_per_node=<NUM_GPUS> rdino_train_ddp.py
```

Training data is not included due to size and privacy constraints.
