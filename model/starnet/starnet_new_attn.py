import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
#import matplotlib
from functools import partial
from model import MODEL
#from model.mobilemamba.wt_function.wavelet_transform import WaveletTransform, InverseWaveletTransform
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import itertools

# --- 假设这些已在代码的其他部分定义 ---
# Conv2d_BN, RepVGGDW, create_wavelet_filter, create_learnable_wavelet_filter,
# wavelet_transform, inverse_wavelet_transform, Residual

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x




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
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

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

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


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

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.GELU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
    





def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    # dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    # dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def create_learnable_wavelet_filter (in_size, out_size, filter_size=2, type=torch.float):
    """
    创建可学习的小波滤波器
    
    Args:
        in_size: 输入通道数
        out_size: 输出通道数
        filter_size: 滤波器大小
        type: 数据类型
    
    Returns:
        learnable_dec_filters: 可学习的分解滤波器
        learnable_rec_filters: 可学习的重构滤波器
    """

    # 初始化可学习滤波器参数
    # 使用正态分布初始化，模拟小波滤波器的特性
    dec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    dec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    
    # 将参数注册为可学习参数
    dec_lo = torch.nn.Parameter(dec_lo)
    dec_hi = torch.nn.Parameter(dec_hi)
    rec_lo = torch.nn.Parameter(rec_lo)
    rec_hi = torch.nn.Parameter(rec_hi)
    
    # 构建滤波器组
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape

    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)

    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape

    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)

    return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class CascadedWTAttn(nn.Module):
    def __init__(self, dim, wt_type='db1', learnable_wavelet=False, stage=0, num_heads=4):
        super(CascadedWTAttn, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.learnable_wavelet = learnable_wavelet
        self.stage = stage

        # === 全局小波变换 ===
        if learnable_wavelet:
            wt_filter, iwt_filter = create_learnable_wavelet_filter(dim, dim, dtype=torch.float)
            self.wt_filter = nn.Parameter(wt_filter)
            self.iwt_filter = nn.Parameter(iwt_filter)
            self.wt_func = partial(wavelet_transform, filters=self.wt_filter)
            self.iwt_func = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        else:
            wt_filter, iwt_filter = create_wavelet_filter(wt_type, dim, dim, torch.float)
            self.wt_func = partial(wavelet_transform, filters=wt_filter)
            self.iwt_func = partial(inverse_wavelet_transform, filters=iwt_filter)

        # === 每 head 独立卷积（作用于 head_dim）===
        self.lh_convs = nn.ModuleList()
        self.hl_convs = nn.ModuleList()
        self.ll_convs = nn.ModuleList()
        for i in range(num_heads):
            self.lh_convs.append(nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim))
            self.hl_convs.append(nn.Conv2d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim))
            kernel = 7 if stage == 0 else (5 if stage == 1 else 3)
            self.ll_convs.append(nn.Conv2d(self.head_dim, self.head_dim, kernel, padding=kernel//2, groups=self.head_dim))

        self.proj = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, dim, 1, bias=False))
        nn.init.zeros_(self.proj[1].weight)

    def forward(self, x):
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        N = H2 * W2

        # === 1. 全局小波变换 ===
        x_wt = self.wt_func(x)  # [B, C, 4, H2, W2]
        ll_full = x_wt[:, :, 0, :, :]
        lh_full = x_wt[:, :, 1, :, :]
        hl_full = x_wt[:, :, 2, :, :]
        hh_full = x_wt[:, :, 3, :, :]

        # === 2. 分组：每个子带拆为 num_heads 组 ===
        ll_groups = list(ll_full.chunk(self.num_heads, dim=1))  # [B, head_dim, H2, W2]
        lh_groups = list(lh_full.chunk(self.num_heads, dim=1))
        hl_groups = list(hl_full.chunk(self.num_heads, dim=1))
        hh_groups = list(hh_full.chunk(self.num_heads, dim=1))

        enhanced_ll_list = []
        prev_out_ll = None

        for i in range(self.num_heads):
            # === 3. 级联输入：原始 ll_groups[i] + 上一个 head 的增强 LL ===
            if i == 0:
                current_ll = ll_groups[i]
            else:
                current_ll = ll_groups[i] + prev_out_ll  # ← 关键：原始第i组 + 上一个输出

            # 获取当前 head 的子带
            lh = lh_groups[i]
            hl = hl_groups[i]
            hh = hh_groups[i]

            # 卷积增强
            lh = self.lh_convs[i](lh)  # Q
            hl = self.hl_convs[i](hl)  # K
            current_ll = self.ll_convs[i](current_ll)  # V

            # === 4. 标准多头注意力（per head，但 head_dim 可视为 single-head）===
            # reshape to [B, head_dim, N]
            q = lh.flatten(2)  # [B, head_dim, N]
            k = hl.flatten(2)  # [B, head_dim, N]
            v = current_ll.flatten(2)  # [B, head_dim, N]

            # Q^T K
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)  # [B, N, N]
            attn = attn.softmax(dim=-1)

            # V @ Attn^T
            out_ll = (v @ attn.transpose(-2, -1)).view(B, self.head_dim, H2, W2)  # [B, head_dim, H2, W2]

            enhanced_ll_list.append(out_ll)
            prev_out_ll = out_ll  # 传递给下一个 head

        # === 5. 拼接所有 head 的输出 ===
        ll_final = torch.cat(enhanced_ll_list, dim=1)  # [B, C, H2, W2]
        lh_final = torch.cat(lh_groups, dim=1)         # 可替换为卷积后 lh
        hl_final = torch.cat(hl_groups, dim=1)
        hh_final = torch.cat(hh_groups, dim=1)

        # === 6. 全局小波逆变换 ===
        wt_final = torch.stack([ll_final, lh_final, hl_final, hh_final], dim=2)  # [B, C, 4, H2, W2]
        output = self.iwt_func(wt_final)  # [B, C, H, W]

        return self.proj(output) + x
    


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()  # Using Hardshrink as PyTorch equivalent to Hardsigmoid

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

class Block(nn.Module):
    def __init__(self, dim, ffn_ratio, drop_path=0., wt_type='db1', learnable_wavelet=False,stage=0,rs = 7,):
        super().__init__()
        self.DW = RepVGGDW(dim)
        self.ese = EffectiveSELayer(dim)
        self.ffn1 = Residual(FFN(dim, int(dim*ffn_ratio)),drop=0)

        self.wtattn = Residual(CascadedWTAttn(dim),drop=0)
        self.ffn2 = Residual(FFN(dim, int(dim * ffn_ratio)),drop=0)
    
    def forward(self, x):
        x_shape = x.shape
        if (x_shape[2] % 2 > 0) or (x_shape[3] % 2 > 0):
            x_pads = (0, x_shape[3] % 2, 0, x_shape[2] % 2)
            x = F.pad(x, x_pads)
        x = self.DW(x)
        x = self.ese(x)
        x = self.ffn1(x)
        x = self.wtattn(x)
        x = self.ffn2(x)
        return x

class FSANet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, dims=[40,80,160,320], depth=[1,2,4,5], mlp_ratio=2., act_layer="GELU",drop_path_rate=0., distillation=False, head_init_scale=0. ,layer_scale_init_value=0., learnable_wavelet=True,down_sample=32):
        super().__init__()
        # if act_layer == "GELU":
        #     act_layer = nn.GELU
        # elif act_layer == "ReLU":
        #     act_layer = nn.ReLU
        # elif act_layer == "Mish":
        #     act_layer = nn.Mish
        resolution = img_size
        self.resolutions = [resolution//8,resolution//16, resolution//32, resolution//64,]
        if down_sample == 32:
            self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1), torch.nn.GELU(),
                                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 1, 1), torch.nn.GELU(),
                                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
                           )
        elif down_sample == 64:
            # self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 8, 3, 2, 1), torch.nn.GELU(),
            #                     Conv2d_BN(dims[0] // 8, dims[0] // 4, 3, 2, 1), torch.nn.GELU(),
            #                     Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1), torch.nn.GELU(),
            #                     Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1), 
            #                )
            self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 8, 3, 2, 1), torch.nn.GELU(),
                            Conv2d_BN(dims[0] // 8, dims[0] // 4, 3, 2, 1), torch.nn.GELU(),
                            Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1), torch.nn.GELU(),
                            Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
                           )
        elif down_sample == 642:
            self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1), torch.nn.GELU(),
                            Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1), torch.nn.GELU(),
                            Conv2d_BN(dims[0] // 2, dims[0] // 1, 3, 2, 1)
                           )
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        for i, (dim, dpth) in enumerate(
                            zip(dims,depth)):
            for j in range(dpth):
                blocks[i].append(Block(dim,ffn_ratio=mlp_ratio, wt_type='db1',drop_path=drop_path_rate, learnable_wavelet=learnable_wavelet,stage=i,rs = self.resolutions[i]))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                blk.append(Conv2d_BN(dims[i], dims[i], ks=3, stride=2, pad=1, groups=dims[i]))
                blk.append(Conv2d_BN(dims[i], dims[i+1], ks=1, stride=1, pad=0))
                
        
        self.head = BN_Linear(dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head(x)
        return x

CFG_StarAttn_T2 = {
        'img_size': 192,
        'dims': [48,96,192,384],
        'depth': [0,1,2,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 32
    }

CFG_StarAttn_T4 = {
        'img_size': 192,
        'dims': [60,120,240,480],
        'depth': [0,1,2,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 32
    }

CFG_StarAttn_T6 = {
        'img_size': 224,
        'dims': [60,120,256,480],
        'depth': [0,1,2,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 32
    }

CFG_StarAttn_T8 = {
        'img_size': 256,
        'dims': [60,120,240,480],
        'depth': [0,2,3,2],
        'drop_path_rate': 0.03,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 32
    }

CFG_StarAttn_T1_64 = {
        'img_size': 192,
        'dims': [72,144,288],
        'depth': [1,2,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
    }

CFG_StarAttn_T2_64 = {
        'img_size': 192,
        'dims': [72,144,288],
        'depth': [2,3,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
    }

CFG_StarAttn_T3_64 = {
        'img_size': 192,
        'dims': [104,208,416],
        'depth': [2,3,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
}

CFG_StarAttn_T4_64 = {
        'img_size': 256,
        'dims': [128,256,512],
        'depth': [2,3,2],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
    }

CFG_StarAttn_T5_64 = {
        'img_size': 256,
        'dims': [128,256,512],
        'depth': [3,4,3],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
    }

CFG_StarAttn_T6_64 = {
        'img_size': 256,
        'dims': [160,320,640],
        'depth': [3,4,3],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 64
    }


CFG_StarAttn_T8_64 = {
        'img_size': 256,
        'dims': [96,192,384,640],
        'depth': [0,3,4,5],
        'drop_path_rate': 0,
        'mlp_ratio': 2,
        "act_layer": "GELU",
        "learnable_wavelet": True,
        "down_sample": 642
    }

# CFG_StarAttn_T8_64 = {
#         'img_size': 256,
#         'dims': [160,320,640],
#         'depth': [3,4,3],
#         'drop_path_rate': 0,
#         'mlp_ratio': 2,
#         "act_layer": "GELU",
#         "learnable_wavelet": True,
#         "down_sample": 64
#     }

# CFG_StarAttn_T6_64 = {
#         'img_size': 256,
#         'dims': [80,160,368,512],
#         'depth': [1,3,4,3],
#         'drop_path_rate': 0,
#         'mlp_ratio': 2,
#         "act_layer": "GELU",
#         "learnable_wavelet": True,
#         "down_sample": 642
#     }


# CFG_StarAttn_T6_64 = {
#         'img_size': 256,
#         'dims': [80,160,384,512],
#         'depth': [1,2,4,5],
#         'drop_path_rate': 0,
#         'mlp_ratio': 2,
#         "act_layer": "GELU",
#         "learnable_wavelet": True,
#         "down_sample": 642
#     }


# CFG_StarAttn_T4_64 = {
#         'img_size': 224,
#         'dims': [64,128,256,512],
#         'depth': [1,2,8,2],
#         'drop_path_rate': 0,
#         'mlp_ratio': 2,
#         "act_layer": "GELU",
#         "learnable_wavelet": True,
#         "down_sample": 64
#     }


# CFG_StarAttn_T6_64 = {
#         'img_size': 224,
#         'dims': [96,192,384,768],
#         'depth': [1,2,8,2],
#         'drop_path_rate': 0,
#         'mlp_ratio': 2,
#         "act_layer": "GELU",
#         "learnable_wavelet": True,
#         "down_sample": 64
#     }


#@MODEL.register_module
#运算量：282.732M, 参数量：4.023M
def FSANet_T2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T2):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：437.306M, 参数量：6.125M
def FSANet_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T4):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：650.977M, 参数量：6.125M
def FSANet_T6(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T6):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：1.023G, 参数量：6.808M
def FSANet_T8(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T8):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model


#@MODEL.register_module
#运算量：82.871M, 参数量：2.121M
def FSANet_64_T1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T1_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：103.652M, 参数量：2.634M
def FSANet_64_T2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T2_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：209.937M, 参数量：5.235M
def FSANet_64_T3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T3_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：297.404M, 参数量：7.698M
def FSANet_64_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T4_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：485.102M, 参数量：11.905M
def FSANet_64_T5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T5_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model

#@MODEL.register_module
#运算量：297.404M, 参数量：7.698M
def FSANet_64_T6(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T6_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model


@MODEL.register_module
#运算量：297.404M, 参数量：7.698M
def FSANet_64_T8_ATTN(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T8_64):
    model = FSANet(num_classes=num_classes, distillation=distillation, **model_cfg)
    return model


#@MODEL.register_module
def StarNet_MHSA_T2_64_DTW_Pre(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_StarAttn_T2_64):
    model = StarNet_MHSA(num_classes=num_classes, distillation=distillation, **model_cfg)
    # weight = torch.load('model_weights/StarNet_MHSA_T2_DTW/net_E.pth')
    # model.load_state_dict(weight, strict=False)
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
    model = FSANet_64_T8()
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3, 256,256).to("cuda")
    # y = model(x)
    # print(y.shape)

    MACs, params = profile(model, inputs=(x,))
    # y = model(x)
    # print(y.shape)
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"运算量：{MACs}, 参数量：{params}")

# --- 使用示例 ---
# 在你的 Block 类中替换 WTAttn:
# 在 Block.__init__ 中:
# self.wtattn = Residual(WTAttn(dim, wt_type=wt_type, learnable_wavelet=learnable_wavelet))
# 替换为:
# self.wtattn = Residual(MHWTAttn(dim, num_heads=4, wt_type=wt_type, learnable_wavelet=learnable_wavelet))
