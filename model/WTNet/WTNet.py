import torch
import itertools

import torch.nn as nn
import torch.nn.functional as F

import pywt
from functools import partial

from .base import Conv2d_BN, Residual, FFN, PatchMerging, BN_Linear


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


class WaveletWindowAttention(nn.Module):
    """
    在频域（小波子带）上执行窗口注意力。
    输入: (B, C, H, W)  # H, W 为原始空间分辨率（如 256）
    内部: 全局 DWT → 频域窗口划分（H//2, W//2）→ 窗口注意力 → 全局 IDWT
    """
    def __init__(self, dim, head_num, wt_type='db1', 
                 resolution=256,          # 原始空间分辨率
                 window_resolution=16,    # 空间域窗口大小（必须为偶数）
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        assert resolution % 2 == 0, "Resolution must be even for DWT"
        assert window_resolution % 2 == 0, "Window resolution must be even"

        self.dim = dim
        self.head_num = head_num
        self.resolution = resolution
        self.window_resolution = window_resolution
        self.freq_window_size = window_resolution // 2  # 频域窗口大小 (e.g., 8)

        # === 可学习小波滤波器 ===
        wt_filter, iwt_filter = create_learnable_wavelet_filter(dim, dim, type=torch.float)
        self.wt_filter = nn.Parameter(wt_filter)
        self.iwt_filter = nn.Parameter(iwt_filter)

        # === 窗口注意力投影头 ===
        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(dim, dim, bn_weight_init=0, resolution=resolution)
        )

        # 每个 head 的 depth-wise 卷积（用于 Q）
        self.dws = nn.ModuleList()
        for i in range(head_num):
            dw = Conv2d_BN(dim//head_num, dim//head_num, kernels[i], 1, kernels[i]//2, 
                          groups=dim//head_num, resolution=resolution)
            self.dws.append(dw)

        # === 相对位置偏置（基于频域窗口） ===
        points = list(itertools.product(
            range(self.freq_window_size), 
            range(self.freq_window_size)
        ))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = nn.Parameter(
            torch.zeros(head_num, len(attention_offsets))
        )
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        self.scale = (dim // head_num) ** -0.5

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def window_partition(self, x, window_size):
        """
        x: (B, C, H, W)
        return: (B * num_windows, C, window_size, window_size)
        """
        B_, C_, H, W = x.shape
        x = x.view(B_, C_, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(-1, C_, window_size, window_size)

    def window_reverse(self, x, window_size, H, W):
        """
        x: (B * num_windows, C, window_size, window_size)
        return: (B, C, H, W)
        """
        Bn = x.shape[0]
        num_windows_h = H // window_size
        num_windows_w = W // window_size
        C = x.shape[1]
        x = x.view(-1, num_windows_h, num_windows_w, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(-1, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.resolution and W == self.resolution, f"Input size {(H, W)} != expected {self.resolution}"

        # === Step 1: 全局小波变换 ===
        x_wt = wavelet_transform(x, filters=self.wt_filter)  # (B, C, 4, H//2, W//2)
        ll = x_wt[:, :, 0, :, :]   # (B, C, H2, W2)
        lh = x_wt[:, :, 1, :, :]
        hl = x_wt[:, :, 2, :, :]
        hh = x_wt[:, :, 3, :, :]
        H2, W2 = H // 2, W // 2

        # === Step 2: 频域窗口划分 ===
        ll_windows = self.window_partition(ll, self.freq_window_size)   # (B*nW, C, win_h, win_w)
        lh_windows = self.window_partition(lh, self.freq_window_size)
        hl_windows = self.window_partition(hl, self.freq_window_size)
        hh_windows = self.window_partition(hh, self.freq_window_size)

        # === Step 3: 窗口内跨频带注意力 ===
        ll_chunks = torch.chunk(ll_windows, self.head_num, dim=1)  # list of (B*nW, C//h, ...)
        lh_chunks = torch.chunk(lh_windows, self.head_num, dim=1)
        hl_chunks = torch.chunk(hl_windows, self.head_num, dim=1)

        lls_out = []
        ll_i = ll_chunks[0]
        for i, dw in enumerate(self.dws):
            if i > 0:
                ll_i = ll_i + ll_chunks[i]  # 累积 LL（保留原设计）

            q = dw(lh_chunks[i])           # (B*nW, C//h, win_h, win_w)
            q = q.flatten(2).transpose(-2, -1)  # (B*nW, N, C//h)
            k = hl_chunks[i].flatten(2)    # (B*nW, C//h, N)
            v = ll_i.flatten(2)            # (B*nW, C//h, N)

            attn = (q @ k) * self.scale
            bias = self.attention_biases[:, self.attention_bias_idxs]  # (h, N, N)
            attn = attn + bias[i].unsqueeze(0)  # 广播到 B*nW
            attn = attn.softmax(dim=-1)

            ll_i = (v @ attn.transpose(-2, -1)).view(
                ll_i.shape[0], -1, self.freq_window_size, self.freq_window_size
            )  # (B*nW, C//h, win_h, win_w)
            lls_out.append(ll_i)

        ll_out_windows = torch.cat(lls_out, dim=1)  # (B*nW, C, win_h, win_w)
        ll_out_windows = self.proj(ll_out_windows)

        # === Step 4: 窗口合并 ===
        ll_out = self.window_reverse(ll_out_windows, self.freq_window_size, H2, W2)  # (B, C, H2, W2)

        # === Step 5: 重建频域图并逆变换 ===
        wt_out = torch.stack([ll_out, lh, hl, hh], dim=2)  # (B, C, 4, H2, W2)
        output = inverse_wavelet_transform(wt_out, filters=self.iwt_filter)  # (B, C, H, W)
        return output


class LocalWindowAttention(nn.Module):
    r"""
    Wrapper that ensures compatibility with existing architectures.
    Now uses WaveletWindowAttention as the core.
    """
    def __init__(self, dim, key_dim=None, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=16,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_resolution = window_resolution

        # 实际注意力模块（频域窗口版）
        self.attn = WaveletWindowAttention(
            dim=dim,
            head_num=num_heads,
            resolution=resolution,
            window_resolution=window_resolution,
            kernels=kernels
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.resolution and W == self.resolution, \
            f"Input size {(H, W)} != expected {self.resolution}"
        return self.attn(x)


class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))


        self.mixer = LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels)
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        x = self.dw0(x)
        x = self.ffn0(x)

        x = self.mixer(x)

        x = self.dw1(x)
        x = self.ffn1(x)
        return x

        # return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))



class WTViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.nh_kd = nh_kd = key_dim * num_heads
#         self.d = int(attn_ratio * key_dim)
#         self.dh = int(attn_ratio * key_dim) * num_heads
#         self.attn_ratio = attn_ratio
#         h = self.dh + nh_kd * 2
#         self.qkv = Conv2d_BN(dim, h, ks=1)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.dh, dim, bn_weight_init=0))
#         self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))

#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]

#     def forward(self, x):
#         B, _, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
#         q = self.dw(q)
#         q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
#         attn = (
#             (q.transpose(-2, -1) @ k) * self.scale
#             +
#             (self.attention_biases[:, self.attention_bias_idxs]
#              if self.training else self.ab)
#         )
#         attn = attn.softmax(dim=-1)
#         x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
#         x = self.proj(x)
#         return x

        
