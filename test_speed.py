"""
Testing the speed of different models
"""
import os
import torch
import torchvision
import time
import timm
#from model.build import EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5
import torchvision
from model import get_model
from argparse import Namespace as _Namespace
#import utils
torch.autograd.set_grad_enabled(False)


T0 = 5
T1 = 10

def fuse_model(model):
    """递归 fuse 模型中的 Conv2d_BN、RepVGGDW、BN_Linear 等模块"""
    from model.starnet.starnet import Conv2d_BN, RepVGGDW, BN_Linear  # 确保导入你的类

    for name, module in model.named_children():
        if isinstance(module, (Conv2d_BN, RepVGGDW, BN_Linear)):
            # 替换为 fuse 后的模块
            fused = module.fuse()
            setattr(model, name, fused)
        else:
            # 递归处理子模块
            fuse_model(module)
    return model

def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    # with torch.cuda.amp.autocast():
    while time.time() - start < T0:
        model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    # with torch.cuda.amp.autocast():
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

if __name__ == "__main__":
    for device in ["cuda:0",'cpu']:

        if 'cuda' in device and not torch.cuda.is_available():
            print("no cuda")
            continue

        if device == 'cpu':
            os.system('echo -n "nb processors "; '
                    'cat /proc/cpuinfo | grep ^processor | wc -l; '
                    'cat /proc/cpuinfo | grep ^"model name" | tail -1')
            print('Using 1 cpu thread')
            torch.set_num_threads(1)
            compute_throughput = compute_throughput_cpu
        else:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            compute_throughput = compute_throughput_cuda
        
        for n, batch_size0, resolution in [
            # ("EfficientViT_M0",1024,256),
            # ('FSANet_64_T1',512,256),
            # ("WTViT_M0",1024,256),
            # ('MobileMamba_T2', 2048, 192),
            # ("MobileMamba_T4", 2048, 192),
            # ('FasterNet_T0',512,224),
            # ('FasterNet_T1',512,224),
            # ('FasterNet_T2',512,224),
            # ('FasterNet_T2',512,224),
            # ("FSANet_64_T6",512,256),
            # ("FSANet_64_T5",512,256),
            # ("FSANet_64_T6",512,256),
            # ("FSANet_64_T8",512,320),
            # ("EfficientViT_M0",512,224),
            # ("EfficientViT_M1",512,224),
            # ("EfficientViT_M2",512,224),
            # ("EfficientViT_M3",512,224),
            # ("EfficientViT_M4",512,224),
            # ("EfficientViT_M5",512,224),

            # ('FSANet_64_T1',512,256),
            # ('FSANet_64_T2',512,256),
            # ('FSANet_64_T2d5',512,256),
            # ('FSANet_64_T3',512,256),
            # ('FSANet_64_T4',512,256),
            # ('FSANet_64_T5',512,256),
            # ('FSANet_64_T6',512,256),
            # ('FSANet_64_T4',512,256),
            # ('StarNet_MHSA_T2_DTW',512,192),
            # ('FSANet_64_T5',512,256),
            # ("StarNet_MHSA_T4_DTW",512,192),
            # ("MobileMamba_T4", 2048, 224),


            # ("starnet_s1", 1024, 224),
            # ("starnet_s2", 1024, 224),
            # ("starnet_s3", 1024, 224),
            # ("starnet_s4", 1024, 224),
            # ("FSANet_64_T1",512,192),
            # ("EfficientViT_M0",512,224),
            # ('mobilenetv4_conv_small_035',512,224),
            # ("FSANet_64_T8",512,256),
            # ("FSANet_64_T8_ATTN",512,256),
            # ('StarNet_MHSA_T2_DTW',512,192),
            # ("StarNet_MHSA_T4_DTW",512,192),
            # ("StarNet_MHSA_T6_DTW",512,192),
            # ('FSANet_64_T2d5',512,256),
            # ('FSANet_64_T2d5_2',512,256),
            # ('FSANet_64_T4',512,256),
            # ('FSANet_64_T5',512,256),
            # ('FSANet_64_T8',1024,256),
            # ('WTConvNeXt_tiny',1024,224),
            # ('CSPConvNeXt_mini',1024,224),
            # ('CSPConvNeXt_tiny',1024,224),
            # ('FSANet_64_T4',1024,256),
            # ('WTConvNeXt_tiny',1024,224),
            ("FSANet_64_T2d5",1024,256),
            # ("FSANet_64_T3",512,256),
            ("FSANet_64_T4",1024,256),
            ("FSANet_CSP_T4",1024,256),
            # ("FSANet_64_T5",512,256),
            # ('FasterNet_T0',512,224),
            # ("FSANet_64_T1",512,256),
            # ("EfficientViT_M0",512,256),
            # ("FSANet_64_T4",512,256),
            # ("EfficientViT_M4",512,224),
            # ("FSANet_64_T5",512,256),
            # ("EfficientViT_M5",512,224),
            # ("FSANet_64_T6",512,320),
            # ("StarNet_MHSA_T8_DTW",1024,256)
            # ("FSANet_64_T6",512,256),
            
            # ('StarNet_MHSA_T6_DTW',512,192),
        
        ]:

            if device == 'cpu':
                batch_size = 16
            else:
                batch_size = batch_size0
                torch.cuda.empty_cache()
            inputs = torch.randn(batch_size, 3, resolution,
                                resolution, device=device)
            torch.cuda.empty_cache()
            model = _Namespace()
            torchvision.models.mobilenet_v3_large(pretrained=False)
            model.name = n
            model.model_kwargs = dict(pretrained=False, checkpoint_path='', ema=False, strict=True, num_classes=1000)
            model = get_model(model)
            model = fuse_model(model)
            model.to(device)
            # model.half()
            model.eval()

            model = torch.jit.trace(model, inputs)
            compute_throughput(n, model, device,
                            batch_size, resolution=resolution)