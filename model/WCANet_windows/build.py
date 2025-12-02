'''
Build the WaveletCenterNet model family (CenterMask + Wavelet + SAG-Mask)
'''
import torch
from model import MODEL
from .WCANet import WaveletCenterNet  # 假设主干网络定义在 wavelet_center_net.py

# ============================
# Model Configurations
# ============================

CFG_WaveletCenterNet_T0 = {
    'img_size': 192,
    'embed_dim': [64, 128, 192],
    'depth': [1, 2, 3],
    'num_heads': [2, 4, 4],
    'window_size': 6,
}

CFG_WaveletCenterNet_T1 = {
    'img_size': 224,
    'embed_dim': [128, 144, 192],
    'depth': [1, 2, 2],
    'num_heads': [2, 3, 3],
    'window_size': 6,
}

CFG_WaveletCenterNet_T2 = {
    'img_size': 256,
    'embed_dim': [128, 192, 224],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 2],
    'window_size': 7,
}

CFG_WaveletCenterNet_T3 = {
    'img_size': 224,
    'embed_dim': [128, 240, 320],
    'depth': [1, 2, 3],
    'num_heads': [4, 3, 4],
    'window_size': 7,
}

CFG_WaveletCenterNet_T4 = {
    'img_size': 224,
    'embed_dim': [160, 320, 640],
    'depth': [3, 4, 5],
    'num_heads': [4, 5, 8],
    'window_size': 7,
}

CFG_WaveletCenterNet_T5 = {
    'img_size': 224,
    'embed_dim': [192, 384, 768],
    'depth': [3, 5, 6],
    'num_heads': [6, 6, 8],
    'window_size': 7,
}


# ============================
# Utility: BatchNorm Fusion
# ============================

def replace_batchnorm(net):
    """Recursively fuse ConvBN and replace BN with Identity."""
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


# ============================
# Model Builders
# ============================

@MODEL.register_module
def WaveletCenterNet_T0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_WaveletCenterNet_T0):
    model = WaveletCenterNet(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available.")
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def WaveletCenterNet_T1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_WaveletCenterNet_T1):
    model = WaveletCenterNet(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available.")
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def WaveletCenterNet_T2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_WaveletCenterNet_T2):
    model = WaveletCenterNet(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available.")
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def WaveletCenterNet_T3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_WaveletCenterNet_T3):
    model = WaveletCenterNet(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available.")
    if fuse:
        replace_batchnorm(model)
    return model


@MODEL.register_module
def WaveletCenterNet_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_WaveletCenterNet_T4):
    model = WaveletCenterNet(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        raise NotImplementedError("Pretrained weights not available.")
    if fuse:
        replace_batchnorm(model)
    return model

