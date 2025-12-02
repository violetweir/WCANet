import torch
import torch.nn as nn
from model import MODEL
from timm.models.layers import trunc_normal_, DropPath


# --------------------- 已提供模块（略作整理） ---------------------
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)



class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(C2f):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super(C2f, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(k, k), e=1.0) for _ in range(n))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


# --------------------- 新增 C2PSA 模块 ---------------------

class PSA(nn.Module):
    """Partial Self-Attention (PSA) module."""
    def __init__(self, c, attn_ratio=0.5, kernel_size=3):
        super().__init__()
        self.c = c
        c_attn = int(c * attn_ratio)
        c_ffn = c - c_attn

        self.attn_conv = nn.Conv2d(c_attn, c_attn, kernel_size, padding=kernel_size//2, groups=c_attn)
        self.attn_bn = nn.BatchNorm2d(c_attn)
        self.attn_act = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c_attn, c_attn, 1)
        self.sigmoid = nn.Sigmoid()

        self.ffn = Conv(c_ffn, c_ffn, 1, 1) if c_ffn > 0 else None

        self.split_channels = (c_attn, c_ffn)

    def forward(self, x):
        x_attn, x_ffn = x.split(self.split_channels, dim=1) if self.split_channels[1] > 0 else (x, None)

        # Attention branch
        a = self.attn_act(self.attn_bn(self.attn_conv(x_attn)))
        a = self.sigmoid(self.fc(self.avg_pool(a)))
        x_attn = x_attn * a

        # FFN branch
        if x_ffn is not None:
            x_ffn = self.ffn(x_ffn)

        return torch.cat([x_attn, x_ffn], dim=1) if x_ffn is not None else x_attn


class C2PSA(C2f):
    """C2f with PSA module."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PSA(self.c) for _ in range(n))


# --------------------- YOLO11 Backbone ---------------------

class YOLO11Backbone(nn.Module):
    def __init__(self, num_classes=1000 ,scale='n'):
        super().__init__()
        # Scaling coefficients
        scales = {
            'n': [0.50, 0.25, 1024],
            's': [0.50, 0.50, 1024],
            'm': [0.50, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
        depth, width, max_channels = scales[scale]

        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)

        c = lambda x: max(make_divisible(x * width, 8), 8)  # channel scaling
        n = lambda x: max(round(x * depth), 1)              # depth scaling

        self.layers = nn.ModuleList([
            # Stage 0: P1/2
            Conv(3, c(64), 3, 2),  # 0
            # Stage 1: P2/4
            Conv(c(64), c(128), 3, 2),  # 1
            # Stage 2: P3/8
            C3k2(c(128), c(256), n(2), False, 0.25),  # 2
            Conv(c(256), c(256), 3, 2),  # 3
            # Stage 3: P4/16
            C3k2(c(256), c(512), n(2), False, 0.25),  # 4
            Conv(c(512), c(512), 3, 2),  # 5
            # Stage 4: P5/32
            C3k2(c(512), c(512), n(2), True),  # 6
            Conv(c(512), c(1024), 3, 2),  # 7
            # Stage 5: top
            C3k2(c(1024), c(1024), n(2), True),  # 8
            SPPF(c(1024), c(1024), 5),  # 9
            C2PSA(c(1024), c(1024), n(2))  # 10
        ])
        self.head = BN_Linear(c(1024), num_classes) if num_classes > 0 else torch.nn.Identity()

        # Adjust channel cap per scale
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                layer.cv2.conv.out_channels = min(layer.cv2.conv.out_channels, max_channels)
            if isinstance(layer, Conv) and layer.conv.out_channels > max_channels:
                layer.conv.out_channels = max_channels
                layer.bn.num_features = max_channels

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in [3, 5, 10]:  # P3, P4, P5 outputs for neck
                outputs.append(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head(x)
        return x  # [P3, P4, P5]


# --------------------- SPPF module (required) ---------------------

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))



@MODEL.register_module
#运算量：1.023G, 参数量：6.808M
def YOLO11s(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None,):
    model = YOLO11Backbone(scale='s',num_classes=num_classes)
    return model

# --------------------- 用法示例 ---------------------

if __name__ == "__main__":
    # model = YOLO11Backbone(scale='s')
    # x = torch.randn(1, 2, 224, 224)

    from thop import profile
    from thop import clever_format
    # model = StarNet_NEW_CONV()
    # x = torch.randn(1, 3, 224, 224).cuda()
    # model = model.cuda()  # Move model to GPU
    # model.eval()
    # y = model(x)
    # print(y.shape)
    # distillation=False
    # pretrained=False
    # num_classes=1000
    # model = StarNet_NEW_CONV()
    # x = torch.randn(1, 3, 224, 224)
    # y = model(x)
    # print(y.shape)
    # print("Model and input are on GPU:", next(model.parameters()).is_cuda)
    # model = StarNet_MHSA(dims=[40,80,160,320], depth=[3, 3, 12, 5], learnable_wavelet=True)
    model = YOLO11Backbone(scale='l')
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3,256,256).to("cuda")
    # y = model(x)
    # print(y.shape)

    MACs, params = profile(model, inputs=(x,))
    # y = model(x)
    # print(y.shape)
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"运算量：{MACs}, 参数量：{params}")