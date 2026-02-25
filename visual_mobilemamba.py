# erf_run_mobilmamba.py

from visualize_erf import compute_erf  # 前面定义的通用 ERF 分析器
from model.mobilemamba.mobilemamba import MobileMambaForERF, CFG_MobileMamba_S6, CFG_MobileMamba_B1, CFG_MobileMamba_B2, CFG_MobileMamba_B4,CFG_MobileMamba_T2, CFG_MobileMamba_T4  # 替换为实际路径

def build_model_for_erf(model_name):
    cfg_map = {
        'MobileMamba_T2': CFG_MobileMamba_T2,
        'MobileMamba_T4': CFG_MobileMamba_T4,
        'MobileMamba_S6': CFG_MobileMamba_S6,
        'MobileMamba_B1': CFG_MobileMamba_B1,
        'MobileMamba_B2': CFG_MobileMamba_B2,
        'MobileMamba_B4': CFG_MobileMamba_B4,
    }
    assert model_name in cfg_map, f"Unsupported model: {model_name}"
    model = MobileMambaForERF(model_cfg=cfg_map[model_name])
    return model

if __name__ == '__main__':
    import argparse
    import torch
    from torchvision import datasets, transforms
    from PIL import Image
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MobileMamba_S6')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='erf_mobilmamba.npy')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=1024)
    args = parser.parse_args()

    # Build ERF-ready model
    model = build_model_for_erf(args.model)
    model.eval()

    # Data loader (resize to input_size)
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=2, pin_memory=True
    )

    # Compute ERF
    erf = compute_erf(model, loader, num_images=args.num_images, device='cuda')
    import numpy as np
    np.save(args.save_path, erf)
    print(f"ERF saved to {args.save_path} with shape {erf.shape}")