import torch
import pywt
import itertools



import torch.nn.functional as F
import torch.nn as nn

from functools import partial


def create_learnable_wavelet_filter(in_size, out_size, filter_size=2, type=torch.float):
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
    # dec_lo = torch.nn.Parameter(dec_lo)
    # dec_hi = torch.nn.Parameter(dec_hi)
    # rec_lo = torch.nn.Parameter(rec_lo)
    # rec_hi = torch.nn.Parameter(rec_hi)
    
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


def window_partition(x, window_size=7):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (B * num_windows, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)

    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B * num_windows, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x



import torch
import torch.nn as nn

class ConvBN(nn.Sequential):
    """
    Conv2d + BatchNorm2d (+ optional activation)
    Matches the style of your existing Conv2d_BN and supports fuse() for deployment.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, act_layer=None, bn_weight_init=1.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        ]
        if act_layer is not None:
            layers.append(act_layer)
        super().__init__(*layers)
        # Initialize BN weight
        torch.nn.init.constant_(self[1].weight, bn_weight_init)
        torch.nn.init.constant_(self[1].bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse Conv + BN into a single Conv2d for inference acceleration.
        Returns a fused nn.Conv2d module.
        """
        conv, bn = self[0], self[1]
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5

        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            device=conv.weight.device
        )
        fused_conv.weight.data.copy_(conv.weight * w[:, None, None, None])
        fused_conv.bias.data.copy_(b)
        return fused_conv

class RepVGGDW(nn.Module):
    def __init__(self, ed):
        super().__init__()
        # 3x3 depth-wise conv + BN (no act)
        self.conv = ConvBN(ed, ed, kernel_size=3, stride=1, padding=1, groups=ed, bias=False)
        # 1x1 point-wise conv + BN (no act)
        self.conv1 = ConvBN(ed, ed, kernel_size=1, stride=1, padding=0, groups=ed, bias=False)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        # Fuse each ConvBN into nn.Conv2d
        conv_fused = self.conv.fuse()      # returns nn.Conv2d with bias
        conv1_fused = self.conv1.fuse()    # returns nn.Conv2d with bias

        # Extract weights and biases
        conv_w = conv_fused.weight          # [C, C, 3, 3] (depth-wise)
        conv_b = conv_fused.bias            # [C]
        conv1_w = conv1_fused.weight        # [C, C, 1, 1] (point-wise)
        conv1_b = conv1_fused.bias          # [C]

        # Pad 1x1 kernel to 3x3
        conv1_w_padded = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])  # [C, C, 3, 3]

        # Build identity kernel (for skip connection)
        identity = torch.zeros_like(conv_w)
        identity[:, :, 1, 1] = 1.0  # [C, C, 3, 3], only center = 1

        # Sum all three branches
        final_weight = conv_w + conv1_w_padded + identity
        final_bias = conv_b + conv1_b  # skip connection has no bias

        # Create fused conv
        fused_conv = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=3, stride=1, padding=1,
            groups=self.dim, bias=True,
            device=final_weight.device
        )
        fused_conv.weight.copy_(final_weight)
        fused_conv.bias.copy_(final_bias)
        return fused_conv


import torch
import torch.nn as nn
import itertools

class HighFreqMHSA(nn.Module):
    """
    Multi-Head Self-Attention for high-frequency features.
    Input: fused_high [B, C, H, W]
    Output: enhanced_high [B, C, H, W]
    """
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.resolution = resolution
        N = resolution * resolution

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # EfficientViT-style relative position bias
        points = list(itertools.product(range(resolution), range(resolution)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.resolution, f"Input resolution {H}x{W} != expected {self.resolution}x{self.resolution}"

        qkv = self.qkv(x)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # each [B, C, H, W]

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)  # [B, heads, N, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, -1)                     # [B, heads, head_dim, N]
        v = v.view(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)   # [B, heads, N, head_dim]

        # Compute attention
        attn = (q @ k) * self.scale  # [B, heads, N, N]
        bias = self.attention_biases[:, self.attention_bias_idxs]  # [heads, N, N]
        attn = attn + bias.unsqueeze(0)  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)

        # Apply attention
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)  # [B, C, H, W]
        return self.proj(out)
    

class LowFreqCrossAttn(nn.Module):
    """
    Cross-Attention: LL (Query) x HighFreq (Key/Value)
    Implements SAG-Mask idea from CenterMask: semantic (LL) guides spatial attention on details.
    """
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.resolution = resolution
        N = resolution * resolution

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Same relative position bias as above
        points = list(itertools.product(range(resolution), range(resolution)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    def forward(self, ll, high_attn):
        B, C, H, W = ll.shape
        assert H == W == self.resolution

        q = self.q_proj(ll)                # [B, C, H, W]
        kv = self.kv_proj(high_attn)       # [B, 2C, H, W]
        k, v = kv.chunk(2, dim=1)          # [B, C, H, W]

        q = q.view(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)  # [B, heads, N, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, -1)                     # [B, heads, head_dim, N]
        v = v.view(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)   # [B, heads, N, head_dim]

        attn = (q @ k) * self.scale
        bias = self.attention_biases[:, self.attention_bias_idxs]
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)
    

class FFN(nn.Module):
    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.pw1 = ConvBN(dim, hidden_dim, 1, act_layer=nn.GELU())
        self.pw2 = ConvBN(hidden_dim, dim, 1, bn_weight_init=0.0)  # 常设为0（残差末尾）

    def forward(self, x):
        return self.pw2(self.pw1(x))

class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)
    
class WaveletDualAttentionBlock(nn.Module):
    """
    CenterMask-inspired block using wavelet subbands.
    
    Forward path:
      High: [lh, hl, hh] → concat → RepVGGDW → PW(3C→C) → HighFreqMHSA → FFN → refined_high
      Low:  ll → eSE → CrossAttn(Q=eSE(ll), KV=refined_high) → FFN -> output
    """
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        # High-frequency fusion & attention
        self.high_fuse_dw = RepVGGDW(ed=dim * 3)
        self.high_fuse_act = nn.GELU()
        self.high_fuse_pw = ConvBN(dim * 3, dim, 1)
        self.high_attn = HighFreqMHSA(dim, num_heads=num_heads, resolution=resolution)
        self.high_mlp = FFN(dim, hidden_ratio=2.0)

        # Low-frequency path
        self.low_ese = EffectiveSELayer(dim)
        self.low_attn = LowFreqCrossAttn(dim, num_heads=num_heads, resolution=resolution)
        self.low_mlp = FFN(dim, hidden_ratio=2.0)

    def forward(self, ll, lh, hl, hh):
        # --- High-frequency path ---
        high_cat = torch.cat([lh, hl, hh], dim=1)          # [B, 3C, H, W]
        high_dw = self.high_fuse_dw(high_cat)              # [B, 3C, H, W]
        high_dw = self.high_fuse_act(high_dw)
        high_compressed = self.high_fuse_pw(high_dw)       # [B, C, H, W]
        high_attn_out = high_compressed+self.high_attn(high_compressed)    # [B, C, H, W]
        refined_high = high_attn_out + self.high_mlp(high_attn_out)        # [B, C, H, W]

        # --- Low-frequency path ---
        ll_ese = self.low_ese(ll)                          # [B, C, H, W]
        ll_attn = ll_ese + self.low_attn(ll_ese, refined_high)      # Q=ll_ese, KV=refined_high
        output = ll_attn + self.low_mlp(ll_attn)                     # final output

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (B * num_windows, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B * num_windows, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

class WindowMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, max_window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_window_size = max_window_size

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        window_size = min(self.max_window_size, H, W)

        # Partition into windows
        x_windows = window_partition(x, window_size)  # [B * N_win, C, ws, ws]
        B_win, _, ws, _ = x_windows.shape

        # QKV
        qkv = self.qkv(x_windows)  # [B*N, 3C, ws, ws]
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention: [B*N, heads, head_dim, ws*ws]
        q = q.view(B_win, self.num_heads, self.head_dim, -1).transpose(-1, -2)  # [B*N, heads, N, head_dim]
        k = k.view(B_win, self.num_heads, self.head_dim, -1)                     # [B*N, heads, head_dim, N]
        v = v.view(B_win, self.num_heads, self.head_dim, -1).transpose(-1, -2)   # [B*N, heads, N, head_dim]

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(-1, -2).view(B_win, C, ws, ws)
        out = self.proj(out)

        # Reverse windowing
        out = window_reverse(out, window_size, H, W)
        return out


class WindowCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=4, max_window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_window_size = max_window_size

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, query, kv):
        """
        query: LL (semantic) → [B, C, H, W]
        kv: refined_high (details) → [B, C, H, W]
        """
        B, C, H, W = query.shape
        window_size = min(self.max_window_size, H, W)

        # Partition both
        q_windows = window_partition(query, window_size)    # [B*N, C, ws, ws]
        kv_windows = window_partition(kv, window_size)      # [B*N, C, ws, ws]
        B_win, _, ws, _ = q_windows.shape

        q = self.q_proj(q_windows)
        kv = self.kv_proj(kv_windows)
        k, v = kv.chunk(2, dim=1)

        q = q.view(B_win, self.num_heads, self.head_dim, -1).transpose(-1, -2)
        k = k.view(B_win, self.num_heads, self.head_dim, -1)
        v = v.view(B_win, self.num_heads, self.head_dim, -1).transpose(-1, -2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(-1, -2).view(B_win, C, ws, ws)
        out = self.proj(out)

        out = window_reverse(out, window_size, H, W)
        return out

class WaveletDualAttentionBlock_Window(nn.Module):
    def __init__(self, dim, num_heads=4, max_window_size=8):
        super().__init__()
        # High-frequency path
        self.high_fuse_dw = RepVGGDW(ed=dim * 3)
        self.high_fuse_pw = ConvBN(dim * 3, dim, 1)
        self.high_attn = WindowMHSA(dim, num_heads=num_heads, max_window_size=max_window_size)
        self.high_mlp = FFN(dim, hidden_ratio=2.0)

        # Low-frequency path
        self.low_dw = RepVGGDW(ed=dim)
        self.low_ese = EffectiveSELayer(dim)
        self.low_attn = WindowCrossAttn(dim, num_heads=num_heads, max_window_size=max_window_size)
        self.low_mlp = FFN(dim, hidden_ratio=2.0)

    def forward(self, ll, lh, hl, hh):
        # High path
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_dw = self.high_fuse_dw(high_cat)
        high_dw = self.high_fuse_act(high_dw)
        high_compressed = self.high_fuse_pw(high_dw)
        high_attn_out = high_compressed + self.high_attn(high_compressed)
        refined_high = high_attn_out + self.high_mlp(high_attn_out)

        # Low path
        ll = self.low_dw(ll)
        ll_ese = self.low_ese(ll)
        ll_attn = ll_ese + self.low_attn(ll_ese, refined_high)
        output = ll_attn + self.low_mlp(ll_attn)
        return output