# erf_analyzer.py

import torch
import numpy as np
from timm.utils import AverageMeter

def get_input_grad(model, samples):
    """
    Compute gradient w.r.t. input at the central output location.
    """
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))  # sum over batch and channel
    return aggregated.cpu().numpy()

def compute_erf(model, data_loader, num_images=50, device='cuda'):
    """
    Compute the average ERF over `num_images` from `data_loader`.
    
    Args:
        model (torch.nn.Module): A model that takes [B, C, H, W] input and outputs [B, C', H', W'].
        data_loader (DataLoader): Must yield (image_tensor, _) with image_tensor shape [1, C, H, W].
        num_images (int): Number of images to average over.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: ERF matrix of shape [H, W], same as input resolution.
    """
    model.to(device)
    model.eval()
    meter = AverageMeter()

    for i, (samples, _) in enumerate(data_loader):
        if meter.count >= num_images:
            break

        samples = samples.to(device, non_blocking=True)
        samples.requires_grad_(True)

        contribution_scores = get_input_grad(model, samples)

        if not np.isnan(np.sum(contribution_scores)):
            meter.update(contribution_scores)
        else:
            print(f"Skipping image {i} due to NaN in gradients.")

    return meter.avg  # shape [H, W]