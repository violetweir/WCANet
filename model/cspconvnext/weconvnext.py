import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import List
from fast_wtconv.wtconv_triton import WTConv2d as WTConv2dTriton
#from model import MODEL


# ========== 工具函数（小波滤波器）==========
def create_2d_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype)
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return self.weight * x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        dec_filter, rec_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(dec_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(rec_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', 
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                      groups=in_channels * 4, bias=False) for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(wt_levels)
        ])

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        x = self.base_scale(self.base_conv(x)) + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x



class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# ========== 原有组件（不变）==========
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def forward(self, x): return x

class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)
    def forward(self, x):
        x_se = x.mean([2, 3], keepdim=True)
        return x * self.act(self.fc(x_se))

class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ========== 修改后的 Block（含 WTConv2d）==========
class Block(nn.Module):
    def __init__(self, dim, wt_levels=1, kernel_size=5, drop_path=0., wt_type='db1'):
        super().__init__()
        # self.wt_conv = WTConv2d(dim, dim, kernel_size=kernel_size, wt_levels=wt_levels, wt_type=wt_type)
        self.wt_conv = WTConv2dTriton(dim, dim, kernel_size=kernel_size, wt_levels=wt_levels).cuda()
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(2 * dim, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ese = EffectiveSELayer(dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.wt_conv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.ese(x)
        x = input + self.drop_path(x * self.gamma)
        return x


# ========== CSPStage（传递 wt_levels）==========
class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, stride, p_rates, wt_levels, kernel_size=5, wt_type='db1'):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        self.down = nn.Sequential(Conv2d_BN(ch_in, ch_in, 3, stride, 1, groups=ch_in), ConvBNLayer(ch_in, ch_mid,1)) if stride == 2 else Identity()
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1)
        self.blocks = nn.Sequential(*[
            Block(ch_mid // 2, wt_levels=wt_levels, kernel_size=kernel_size, drop_path=p_rates[i], wt_type=wt_type)
            for i in range(n)
        ])
        self.attn = EffectiveSELayer(ch_mid)
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1)

    def forward(self, x):
        x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        y = self.attn(y)
        return self.conv3(y)


# ========== 主干模型 ==========
class CSPConvNeXt(nn.Module):
    # arch_settings = {
    #     'mini':  {'depths': [3,3,9,3], 'dims': [48,96,192,384,768], 'stem': 'va', 'stride': [1,2,2,2]},
    #     'tiny':  {'depths': [3,3,9,3], 'dims': [64,128,256,512,1024], 'stem': 'vb', 'stride': [2,2,2,2]},
    #     'small': {'depths': [3,3,27,3], 'dims': [96,192,384,768,768], 'stem': 'vb', 'stride': [2,2,2,2]},
    # }
    arch_settings = {
        'mini':  {'depths': [3,3,9,3], 'dims': [48,96,192,384,768], 'stem': 'va', 'stride': [1,2,2,2]},
        'tiny':  {'depths': [0,2,3,2], 'dims': [32,64,128,384,448], 'stem': 'vb', 'stride': [2,2,2,2]},
        'small': {'depths': [3,3,27,3], 'dims': [64,128,256,512,1024], 'stem': 'vb', 'stride': [2,2,2,2]},
    }

    def __init__(self, arch='tiny', in_chans=3, drop_path_rate=0., class_num=1000,
                 kernel_size=5, wt_type='db1', depth_mult=1.0, width_mult=1.0):
        super().__init__()
        cfg = self.arch_settings[arch]
        depths = [int(d * depth_mult) for d in cfg['depths']]
        dims   = [int(d * width_mult) for d in cfg['dims']]
        stem_type = cfg['stem']
        strides   = cfg['stride']

        # Stem
        if stem_type == 'va':
            self.Down_Conv = nn.Sequential(
                ConvBNLayer(in_chans, (dims[0]+dims[1])//2, 4, 4, 0)
            )
        else:
            self.Down_Conv = nn.Sequential(
                ConvBNLayer(in_chans, dims[0]//4, 3, 2, 1),
                ConvBNLayer(dims[0]//4, dims[0]//2, 3, 1, 1),
                ConvBNLayer(dims[0]//2, dims[0], 3, 2, 1)
            )

        # Stages with WT levels: [5,4,3,2]
        wt_levels_list = [5,4,3,2]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages = []
        for i in range(4):
            stages.append(CSPStage(
                ch_in=dims[i],
                ch_out=dims[i+1],
                n=depths[i],
                stride=strides[i],
                p_rates=dp_rates[sum(depths[:i]):sum(depths[:i+1])],
                wt_levels=wt_levels_list[i],
                kernel_size=kernel_size,
                wt_type=wt_type
            ))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(dims[-1], class_num)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Down_Conv(x)
        for stage in self.stages:
            x = stage(x)
        print(x.shape)
        x = x.mean([-2, -1])  # global avg pool
        return self.head(x)


# ========== 快捷构造函数 ==========
def e_convnext_mini_wt(**kwargs):
    return CSPConvNeXt(arch='mini', **kwargs)

def e_convnext_tiny_wt(**kwargs):
    return CSPConvNeXt(arch='tiny', **kwargs)

def e_convnext_small_wt(**kwargs):
    return CSPConvNeXt(arch='small', **kwargs)

#@MODEL.register_module
#运算量：82.871M, 参数量：2.121M
def WTConvNeXt_tiny(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    model = CSPConvNeXt(arch='tiny', class_num=num_classes )
    return model

#@MODEL.register_module
#运算量：82.871M, 参数量：2.121M
def WTConvNeXt_small(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    model = CSPConvNeXt(arch='small', class_num=num_classes )
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
    model = e_convnext_tiny_wt(class_num=1000)
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
