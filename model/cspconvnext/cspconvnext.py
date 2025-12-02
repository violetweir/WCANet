import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
import math
from model import MODEL


# ==============================
# Utils
# ==============================

def _trunc_normal_(tensor, mean=0., std=1.):
    # Based on https://github.com/pytorch/pytorch/issues/29883
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # For compatibility; use approximate version above
    _trunc_normal_(tensor, mean, std)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ==============================
# Layers
# ==============================

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation (eSE) from CenterMask / VoVNetV2 """
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)  # global average pooling
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        size: int,  # unused in current implementation, keep for compatibility
        kernel_size: int = 7,
        if_group: int = 1,
        drop_path: float = 0.,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        groups = dim if if_group == 1 else 1
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ese = EffectiveSELayer(dim)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.ese(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = input + self.drop_path(x)
        return x


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        filter_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=filter_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.GELU() if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class CSPStage(nn.Module):
    def __init__(
        self,
        block_fn,
        ch_in: int,
        ch_out: int,
        n: int,
        stride: int,
        p_rates: List[float],
        size: int,
        kernel_size: int = 7,
        if_group: int = 1,
        layer_scale_init_value: float = 1e-6,
        attn: bool = True  # enable eSE at stage output
    ):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.down = ConvBNLayer(ch_in, ch_mid, filter_size=2, stride=2, padding=0)
        else:
            self.down = Identity()

        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, filter_size=1)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, filter_size=1)

        blocks = []
        for i in range(n):
            blocks.append(
                block_fn(
                    dim=ch_mid // 2,
                    size=size,
                    kernel_size=kernel_size,
                    if_group=if_group,
                    drop_path=p_rates[i],
                    layer_scale_init_value=layer_scale_init_value
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.attn = EffectiveSELayer(ch_mid) if attn else None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, filter_size=1)

    def forward(self, x):
        x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


# ==============================
# Main Model
# ==============================

class ClasHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1000):
        super().__init__()
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, C] after global pooling
        return self.fc_cls(x)


class CSPConvNeXt(nn.Module):
    arch_settings = {
        'mini': {
            'depths': [3, 3, 9, 3],
            'dims': [48, 96, 192, 384, 768],
            'stem': 'va',
            'stride': [1, 2, 2, 2]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [64, 128, 256, 512, 1024],
            'stem': 'vb',
            'stride': [2, 2, 2, 2]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [64, 128, 256, 512, 1024],
            'stem': 'vb',
            'stride': [2, 2, 2, 2]
        },
    }

    def __init__(
        self,
        arch: str = 'tiny',
        in_chans: int = 3,
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        class_num: int = 1000,
        kernel_size: int = 7,
        if_group: int = 1,
        depth_mult: float = 1.0,
        width_mult: float = 1.0,
        **kwargs
    ):
        super().__init__()
        assert arch in self.arch_settings, f"arch {arch} not in {list(self.arch_settings.keys())}"
        cfg = self.arch_settings[arch]
        depths = [int(d * depth_mult) for d in cfg['depths']]
        dims = [int(d * width_mult) for d in cfg['dims']]
        stem_type = cfg['stem']
        strides = cfg['stride']

        # Stem
        act = True  # GELU
        if stem_type == "va":
            self.Down_Conv = nn.Sequential(
                ConvBNLayer(in_chans, (dims[0] + dims[1]) // 2, filter_size=4, stride=4, padding=0, act=act)
            )
        else:  # "vb"
            self.Down_Conv = nn.Sequential(
                ConvBNLayer(in_chans, dims[0] // 2, filter_size=2, stride=2, padding=0, act=act),
                ConvBNLayer(dims[0] // 2, dims[0] // 2, filter_size=3, stride=1, padding=1, act=act),
                ConvBNLayer(dims[0] // 2, dims[0], filter_size=3, stride=2, padding=1, act=act),
            )

        # Stochastic depth
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Sizes (not used in forward, but kept for compatibility)
        sizes = [224 // 4, 224 // 8, 224 // 16, 224 // 32]

        stages = []
        for i in range(len(depths)):
            stage = CSPStage(
                block_fn=Block,
                ch_in=dims[i],
                ch_out=dims[i + 1],
                n=depths[i],
                stride=strides[i],
                p_rates=dp_rates[sum(depths[:i]):sum(depths[:i+1])],
                size=sizes[i],
                kernel_size=kernel_size,
                if_group=if_group,
                layer_scale_init_value=layer_scale_init_value,
                attn=True
            )
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        self.head = ClasHead(in_channels=dims[-1], num_classes=class_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Down_Conv(x)
        for stage in self.stages:
            x = stage(x)
        # Global average pooling
        x = x.mean(dim=[-2, -1])  # [B, C, H, W] -> [B, C]
        x = self.head(x)
        return x


# ==============================
# Convenience constructors
# ==============================

def e_convnext_mini(**kwargs):
    return CSPConvNeXt(arch='mini', **kwargs)

def e_convnext_tiny(**kwargs):
    return CSPConvNeXt(arch='tiny', **kwargs)

def e_convnext_small(**kwargs):
    return CSPConvNeXt(arch='small', **kwargs)


@MODEL.register_module
#运算量：82.871M, 参数量：2.121M
def CSPConvNeXt_tiny(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    model = CSPConvNeXt(arch='tiny', class_num=num_classes )
    return model


if __name__ == "__main__":
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
    model = e_convnext_tiny(class_num=1000)
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3,224,224).to("cuda")
    # y = model(x)
    # print(y.shape)

    MACs, params = profile(model, inputs=(x,))
    # y = model(x)
    # print(y.shape)
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"运算量：{MACs}, 参数量：{params}")

