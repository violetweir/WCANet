import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import List
from model import MODEL


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


# ========== 基础模块 ==========
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:],
            stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups,
            device=c.weight.device, bias=True
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
        identity = torch.eye(self.dim, device=conv1_w.device).view(self.dim, self.dim, 1, 1)
        identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = Conv2d_BN(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x_se = x.mean([2, 3], keepdim=True)
        return x * self.act(self.fc(x_se))


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.GELU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return self.weight * x

# ========== 小波卷积模块（使用 Conv2d_BN）==========
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # Replace wavelet_convs with RepVGGDW blocks
        self.wavelet_blocks = nn.ModuleList(
            [RepVGGDW(in_channels * 4) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            # Apply RepVGGDW, then scale
            curr_x_tag = self.wavelet_blocks[i](curr_x_tag)
            curr_x_tag = self.wavelet_scale[i](curr_x_tag)
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# ========== 安全下采样（使用 Conv2d_BN）==========
class SafeDownsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = Conv2d_BN(ch_in, ch_out, ks=2, stride=2, pad=0)
        self.act = nn.GELU()

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.conv(x)
        x = self.act(x)
        return x


# ========== 新 Block 结构 ==========
class Block(nn.Module):
    def __init__(self, dim, wt_levels=1, kernel_size=5, drop_path=0., wt_type='db1'):
        super().__init__()
        
        # 1. RepVGGDW → ESE
        self.rep_ese = nn.Sequential(
            RepVGGDW(dim),
            EffectiveSELayer(dim)
        )

        # 2. First FFN
        self.ffn1 = Residual(FFN(dim, dim * 2), drop=drop_path)

        # 3. WTConv2d
        self.wt_conv = Residual(
            WTConv2d(dim, dim, kernel_size=kernel_size, wt_levels=wt_levels, wt_type=wt_type),
            drop=drop_path
        )

        # 4. Second FFN
        self.ffn2 = Residual(FFN(dim, dim * 2), drop=drop_path)

    def forward(self, x):
        x = self.rep_ese(x)
        x = self.ffn1(x)
        x = self.wt_conv(x)
        x = self.ffn2(x)
        return x


# ========== CSPStage（使用新 Block）==========
class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, stride, p_rates, wt_levels, kernel_size=5, wt_type='db1'):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.down = SafeDownsample(ch_in, ch_mid)
        else:
            self.down = nn.Identity()

        self.conv1 = nn.Sequential(
            Conv2d_BN(ch_mid, ch_mid // 2, 1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            Conv2d_BN(ch_mid, ch_mid // 2, 1),
            nn.GELU()
        )

        self.blocks = nn.Sequential(*[
            Block(ch_mid // 2, wt_levels=wt_levels, kernel_size=kernel_size,
                  drop_path=p_rates[i], wt_type=wt_type)
            for i in range(n)
        ])

        self.attn = EffectiveSELayer(ch_mid)

        self.conv3 = nn.Sequential(
            Conv2d_BN(ch_mid, ch_out, 1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        y = self.attn(y)
        y = self.conv3(y)
        return y


# ========== 主干模型 ==========
class CSPConvNeXt(nn.Module):
    arch_settings = {
        'mini':  {'depths': [3,3,9,3], 'dims': [48,96,192,384,768], 'stem': 'va', 'stride': [1,2,2,2]},
        'tiny':  {'depths': [2,3,4,5], 'dims': [64,128,256,512,1024], 'stem': 'vb', 'stride': [2,2,2,2]},
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
                Conv2d_BN(in_chans, (dims[0] + dims[1]) // 2, 4, stride=4),
                nn.GELU()
            )
        else:
            self.Down_Conv = nn.Sequential(
                Conv2d_BN(in_chans,dims[0]//2, 2, 2, 0),nn.GELU(),
                Conv2d_BN(dims[0]//2,dims[0]//2, 3, 2, 1, groups=dims[0]//2), 
                Conv2d_BN(dims[0]//2,dims[0], 1, 1, 0,), nn.GELU()
            )

        wt_levels_list = [5, 4, 3, 2]
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
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Down_Conv(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean([-2, -1])  # global avg pool
        return self.head(x)


# ========== 快捷构造函数 ==========
def e_convnext_mini_wt(**kwargs):
    return CSPConvNeXt(arch='mini', **kwargs)

def e_convnext_tiny_wt(**kwargs):
    return CSPConvNeXt(arch='tiny', **kwargs)

def e_convnext_small_wt(**kwargs):
    return CSPConvNeXt(arch='small', **kwargs)
@MODEL.register_module
def WTConvNeXt_tiny(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    model = CSPConvNeXt(arch='tiny', class_num=num_classes)
    return model


# ========== 测试脚本 ==========
if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    model = e_convnext_tiny_wt(class_num=1000)
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3, 224, 224).to("cuda")

    MACs, params = profile(model, inputs=(x,))
    MACs, params = clever_format([MACs, params], '%.3f')
    print(f"运算量：{MACs}, 参数量：{params}")