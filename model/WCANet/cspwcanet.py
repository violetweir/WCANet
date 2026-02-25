import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from functools import partial
from typing import List
#from model import MODEL  # 若无此模块，可注释掉 @MODEL.register_module

# ==============================
# 工具函数
# ==============================

def _trunc_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    _trunc_normal_(tensor, mean, std)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob or 0.0
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

# ==============================
# 基础组件（来自原 FSANet）
# ==============================

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.c.stride,
            padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device
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

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop
    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.GELU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)
    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))

# ==============================
# 小波相关
# ==============================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device="cuda").flip(dims=[0])
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
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTAttn(nn.Module):
    def __init__(self, dim, wt_type='db1', learnable_wavelet=False, stage=0):
        super(WTAttn, self).__init__()
        self.learnable_wavelet = learnable_wavelet
        
        if learnable_wavelet:
            wt_filter, iwt_filter = create_learnable_wavelet_filter(dim, dim, type=torch.float)
            self.wt_filter = nn.Parameter(wt_filter)
            self.iwt_filter = nn.Parameter(iwt_filter)
            self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
            self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        else:
            wt_filter, iwt_filter = create_wavelet_filter(wt_type, dim, dim, torch.float)
            self.wt_function = partial(wavelet_transform, filters=wt_filter)
            self.iwt_function = partial(inverse_wavelet_transform, filters=iwt_filter)

        self.lh_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.hl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        if stage == 0:
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        elif stage == 1:
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        else:
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        if (x.shape[2] % 2 != 0) or (x.shape[3] % 2 != 0):
            x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2))
        x_wt = self.wt_function(x)
        ll, lh, hl, hh = x_wt[:, :, 0], x_wt[:, :, 1], x_wt[:, :, 2], x_wt[:, :, 3]

        lh_conv = self.lh_conv(lh)
        hl_conv = self.hl_conv(hl)
        ll_conv = self.ll_conv(ll)
        attn = (self.act(lh_conv * hl_conv) * ll_conv) + ll
        wt_map = torch.stack([attn, lh_conv, hl_conv, hh], dim=2)
        output = self.iwt_function(wt_map)
        return output

class EffectiveSELayer(nn.Module):
    """ From CenterMask (arXiv:1911.06667) """
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

# ==============================
# 新 Block：融合 WTAttn
# ==============================

class FSABlock(nn.Module):
    def __init__(self, dim, ffn_ratio=2., drop_path=0., wt_type='db1', learnable_wavelet=False, stage=0):
        super().__init__()
        self.dw = RepVGGDW(dim)
        self.ese = EffectiveSELayer(dim)
        self.ffn1 = Residual(FFN(dim, int(dim * ffn_ratio)), drop=0)
        self.wtattn = Residual(WTAttn(dim, wt_type=wt_type, learnable_wavelet=learnable_wavelet, stage=stage), drop=0)
        self.ffn2 = Residual(FFN(dim, int(dim * ffn_ratio)), drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.ese(x)
        x = self.ffn1(x)
        x = self.wtattn(x)
        x = self.ffn2(x)
        x = identity + self.drop_path(x)
        return x

# ==============================
# CSPStage（来自 CSPConvNeXt）
# ==============================

class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, padding=0, groups=1, act=True):
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
        attn: bool = True
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
            blocks.append(block_fn(dim=ch_mid // 2))
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
# 主模型：FSANet_CSP
# ==============================

class FSANet_CSP(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        dims=[48, 96, 192, 384],      # stem 输出 + 各 stage 输出（长度应为 4）
        depth=[1, 2, 4, 5],
        mlp_ratio=2.,
        drop_path_rate=0.,
        learnable_wavelet=True,
        down_sample=32,
        kernel_size=7,
        if_group=1,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        # Patch Embedding
        if down_sample == 32:
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1),
                nn.GELU(),
                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 1, 1),
                nn.GELU(),
                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
            )
        elif down_sample == 64:
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_chans, dims[0] // 8, 3, 2, 1),
                nn.GELU(),
                Conv2d_BN(dims[0] // 8, dims[0] // 4, 3, 2, 1),
                nn.GELU(),
                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1),
                nn.GELU(),
                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
            )
        else:
            raise ValueError("down_sample must be 32 or 64")

        # Expand dims to [stem, s1, s2, s3, s4]
        full_dims = [dims[0]] + dims  # e.g., [48, 48, 96, 192, 384]
        total_depth = sum(depth)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depth)):
            ch_in = full_dims[i]
            ch_out = full_dims[i + 1]
            n = depth[i]
            stride = 2 if i > 0 else 1

            stage = CSPStage(
                block_fn=lambda dim, idx=i: FSABlock(
                    dim=dim,
                    ffn_ratio=mlp_ratio,
                    drop_path=dp_rates[cur + idx] if cur + idx < len(dp_rates) else 0.,
                    wt_type='db1',
                    learnable_wavelet=learnable_wavelet,
                    stage=i
                ),
                ch_in=ch_in,
                ch_out=ch_out,
                n=n,
                stride=stride,
                p_rates=dp_rates[cur:cur + n],
                size=img_size // (4 * (2 ** i)),
                kernel_size=kernel_size,
                if_group=if_group,
                layer_scale_init_value=layer_scale_init_value,
                attn=True
            )
            self.stages.append(stage)
            cur += n

        self.head = BN_Linear(full_dims[-1], num_classes) if num_classes > 0 else Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        print(x.shape)
        for stage in self.stages:
            x = stage(x)
            print(x.shape)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head(x)
        return x

# ==============================
# 配置与构造函数
# ==============================

CFG_FSANet_CSP_T4 = {
    'img_size': 256,
    'dims': [128,256,512],
    'depth': [2,3,2],
    'drop_path_rate': 0.03,
    'mlp_ratio': 2,
    'learnable_wavelet': True,
    'down_sample': 64
}

#@MODEL.register_module
def FSANet_CSP_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    cfg = CFG_FSANet_CSP_T4.copy()
    # Adjust dims: insert stem output at front
    dims = cfg.pop('dims')
    cfg['dims'] = dims  # [60, 60, 120, 240, 480]
    model = FSANet_CSP(num_classes=num_classes, **cfg)
    return model

# ==============================
# 测试
# ==============================

if __name__ == "__main__":
    from thop import profile, clever_format
    model = FSANet_CSP_T4(num_classes=1000)
    model.eval()
    model.to("cuda")
    x = torch.randn(1, 3, 256,256).to("cuda")
    MACs, params = profile(model, inputs=(x,))
    MACs, params = clever_format([MACs, params], '%.3f')
    print(f"运算量：{MACs}, 参数量：{params}")