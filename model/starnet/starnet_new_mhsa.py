import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from functools import partial
from model import MODEL
from timm.models.layers import trunc_normal_

# ======================
# 基础组件
# ======================

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
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0),
                            w.shape[2:], stride=self.c.stride,
                            padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups, device=c.weight.device)
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
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + l.bias
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


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


# ======================
# 小波工具函数
# ======================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def create_learnable_wavelet_filter(in_size, out_size, filter_size=2, type=torch.float):
    dec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    dec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1

    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    x = x.reshape(b, c * 4, h_half, w_half)
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


# ======================
# Wavelet Transformer（全局，无窗口）
# ======================

class WaveletTransformer(nn.Module):
    def __init__(self, dim, head_num=4, wt_type='db1', learnable_wavelet=True):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.learnable_wavelet = learnable_wavelet

        if learnable_wavelet:
            wt_filter, iwt_filter = create_learnable_wavelet_filter(dim, dim, type=torch.float)
            self.wt_filter = nn.Parameter(wt_filter)
            self.iwt_filter = nn.Parameter(iwt_filter)
        else:
            wt_filter, iwt_filter = create_wavelet_filter(wt_type, dim, dim, torch.float)
            self.register_buffer('wt_filter', wt_filter)
            self.register_buffer('iwt_filter', iwt_filter)

        self.dws = nn.ModuleList()
        self.scale = (dim // head_num) ** -0.5
        for _ in range(head_num):
            dw = nn.Conv2d(dim // head_num, dim // head_num,
                           kernel_size=3, padding=1, groups=dim // head_num)
            self.dws.append(dw)

        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(dim, dim, bn_weight_init=0)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))

        # DWT
        if self.learnable_wavelet:
            x_wt = wavelet_transform(x, self.wt_filter)
        else:
            x_wt = wavelet_transform(x, self.wt_filter)
        ll = x_wt[:, :, 0, :, :]
        lh = x_wt[:, :, 1, :, :]
        hl = x_wt[:, :, 2, :, :]
        hh = x_wt[:, :, 3, :, :]

        # Chunk by heads
        ll_chunks = torch.chunk(ll, self.head_num, dim=1)
        lh_chunks = torch.chunk(lh, self.head_num, dim=1)
        hl_chunks = torch.chunk(hl, self.head_num, dim=1)

        lls_out = []
        ll_i = ll_chunks[0]
        for i, dw in enumerate(self.dws):
            if i > 0:
                ll_i = ll_i + ll_chunks[i]
            q = dw(lh_chunks[i])
            q = q.flatten(2).transpose(-2, -1)
            k = hl_chunks[i].flatten(2)
            v = ll_i.flatten(2)
            attn = (q @ k) * self.scale
            attn = attn.softmax(dim=-1)
            ll_i = (v @ attn.transpose(-2, -1)).view(B, -1, H // 2, W // 2)
            lls_out.append(ll_i)

        ll_out = torch.cat(lls_out, dim=1)
        ll_out = self.proj(ll_out)
        wt_out = torch.stack([ll_out, lh, hl, hh], dim=2)

        if self.learnable_wavelet:
            output = inverse_wavelet_transform(wt_out, self.iwt_filter)
        else:
            output = inverse_wavelet_transform(wt_out, self.iwt_filter)

        return output


# ======================
# Block & WTNet
# ======================

class Block(nn.Module):
    def __init__(self, dim, ffn_ratio, drop_path=0., wt_type='db1',
                 learnable_wavelet=True, stage=0):
        super().__init__()
        self.DW = RepVGGDW(dim)
        self.ese = EffectiveSELayer(dim)
        self.ffn1 = Residual(FFN(dim, int(dim * ffn_ratio)), drop=0)
        self.wtattn = Residual(
            WaveletTransformer(dim, head_num=4, wt_type=wt_type,
                               learnable_wavelet=learnable_wavelet), drop=0
        )
        self.ffn2 = Residual(FFN(dim, int(dim * ffn_ratio)), drop=0)

    def forward(self, x):
        x_shape = x.shape
        if (x_shape[2] % 2 > 0) or (x_shape[3] % 2 > 0):
            x = F.pad(x, (0, x_shape[3] % 2, 0, x_shape[2] % 2))
        x = self.DW(x)
        x = self.ese(x)
        x = self.ffn1(x)
        x = self.wtattn(x)
        x = self.ffn2(x)
        return x


class WTNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 dims=[48, 96, 192, 384], depth=[1, 2, 4, 5],
                 mlp_ratio=2., drop_path_rate=0.,
                 learnable_wavelet=True, down_sample=32):
        super().__init__()

        # Patch embedding
        if down_sample == 32:
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1), nn.GELU(),
                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 1, 1), nn.GELU(),
                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
            )
        elif down_sample == 64:
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_chans, dims[0] // 8, 3, 2, 1), nn.GELU(),
                Conv2d_BN(dims[0] // 8, dims[0] // 4, 3, 2, 1), nn.GELU(),
                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1), nn.GELU(),
                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
            )
        else:
            raise ValueError("down_sample must be 32 or 64")

        # Build stages
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]

        for i, (dim, dpth) in enumerate(zip(dims, depth)):
            for j in range(dpth):
                blocks[i].append(Block(dim, ffn_ratio=mlp_ratio,
                                       wt_type='db1',
                                       learnable_wavelet=learnable_wavelet,
                                       stage=i))
            if i != len(depth) - 1:
                blk = blocks[i + 1]
                blk.append(Conv2d_BN(dims[i], dims[i], ks=3, stride=2, pad=1, groups=dims[i]))
                blk.append(Conv2d_BN(dims[i], dims[i + 1], ks=1, stride=1, pad=0))

        self.head = BN_Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head(x)
        return x


# ======================
# 配置 & 工厂函数
# ======================

CFG_WTNet_T1_64 = {
        'img_size': 256,
        'dims': [64,128,256],
        'depth': [2,3,2],
        'mlp_ratio': 2,
        "learnable_wavelet": True,
        "down_sample": 64 
    }

@MODEL.register_module
def WTNet_T1_64(num_classes=1000, **kwargs):
    return WTNet(num_classes=num_classes, **CFG_WTNet_T1_64)


CFG_WTNet_T4_64 = {
    'img_size': 256,
    'dims': [128, 256, 512],
    'depth': [2, 3, 2],
    'mlp_ratio': 2,
    'learnable_wavelet': True,
    'down_sample': 64
}

@MODEL.register_module
def WTNet_T4_64(num_classes=1000, **kwargs):
    return WTNet(num_classes=num_classes, **CFG_WTNet_T4_64)


# ======================
# 测试示例
# ======================

if __name__ == "__main__":
    from thop import profile, clever_format
    model = WTNet_T4_64(num_classes=1000)
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3, 256, 256).to("cuda")
    y = model(x)
    print("Output shape:", y.shape)

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], '%.3f')
    print(f"运算量：{macs}, 参数量：{params}")