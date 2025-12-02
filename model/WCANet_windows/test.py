import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from build import (
    WaveletCenterNet_T0,
    WaveletCenterNet_T1,
    WaveletCenterNet_T2,
    WaveletCenterNet_T3,
    WaveletCenterNet_T4,
    WaveletCenterNet_T5,
)

# Model configs (must match the input size)
MODELS = [
    ("WaveletCenterNet_T0", WaveletCenterNet_T0, 256),
    ("WaveletCenterNet_T1", WaveletCenterNet_T1, 256),
    ("WaveletCenterNet_T2", WaveletCenterNet_T2, 256),
    ("WaveletCenterNet_T3", WaveletCenterNet_T3, 256),
    ("WaveletCenterNet_T4", WaveletCenterNet_T4, 256),
    ("WaveletCenterNet_T5", WaveletCenterNet_T5, 256),
]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'Model':<25} | {'Resolution':<12} | {'FLOPs (G)':<12} | {'Params (M)':<12}")
    print("-" * 70)

    for name, model_fn, img_size in MODELS:
        # Build model
        model = model_fn(num_classes=1000)
        model.eval()
        model.to(device)

        # Dummy input
        x = torch.randn(1, 3, img_size, img_size).to(device)

        # FLOPs
        flops = FlopCountAnalysis(model, x)
        flops_g = flops.total() / 1e9

        # Params
        params = parameter_count(model)['']
        params_m = params / 1e6

        print(f"{name:<25} | {img_size}x{img_size:<8} | {flops_g:<12.2f} | {params_m:<12.2f}")