import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from fvcore.nn import FlopCountAnalysis, parameter_count

# ============================
# 1. 基础组件
# ============================

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 groups=1, act_layer=None, bn_weight_init=1.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act_layer is not None:
            layers.append(act_layer)
        super().__init__(*layers)
        nn.init.constant_(self[1].weight, bn_weight_init)
        nn.init.constant_(self[1].bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        conv, bn = self[0], self[1]
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          conv.stride, conv.padding, conv.groups, bias=True)
        fused.weight.copy_(conv.weight * w[:, None, None, None])
        fused.bias.copy_(b)
        return fused


class EffectiveSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid()
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        return x * self.act(self.fc(x_se))


class FFN(nn.Module):
    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.pw1 = ConvBN(dim, hidden_dim, 1, act_layer=nn.GELU())
        self.pw2 = ConvBN(hidden_dim, dim, 1, bn_weight_init=0.0)
    def forward(self, x):
        return self.pw2(self.pw1(x))


# ============================
# 2. 带位置编码的全局注意力（仿 CascadedGroupAttention）
# ============================

class GlobalMHSAWithPosBias(nn.Module):
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.resolution = resolution
        self.qkv = ConvBN(dim, dim * 3, 1, act_layer=None)
        self.proj = nn.Sequential(nn.GELU(),ConvBN(dim, dim, 1, act_layer=None))
        # Build relative position bias table
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
        self.num_relative_distances = len(attention_offsets)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.num_relative_distances))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.resolution and W == self.resolution, \
            f"Input size ({H}x{W}) != initialized resolution ({self.resolution}x{self.resolution})"
        N = H * W
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(-1, -2)
        attn = (q @ k) * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).view(B, C, H, W)
        return self.proj(out)


class GlobalCrossAttnWithPosBias(nn.Module):
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.resolution = resolution
        self.q_proj = ConvBN(dim, dim, 1, act_layer=None)
        self.kv_proj = ConvBN(dim, 2 * dim, 1, act_layer=None)
        self.proj = nn.Sequential(nn.GELU(),ConvBN(dim, dim, 1, act_layer=None))

        # Position bias for query
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
        self.num_relative_distances = len(attention_offsets)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.num_relative_distances))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, query, kv):
        B, C, H, W = query.shape
        assert H == self.resolution and W == self.resolution, \
            f"Input size ({H}x{W}) != initialized resolution ({self.resolution}x{self.resolution})"
        N = H * W
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        kv = self.kv_proj(kv).view(B, 2, self.num_heads, self.head_dim, N)
        k, v = kv[:, 0], kv[:, 1]
        attn = (q @ k) * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).view(B, C, H, W)
        return self.proj(out)


# ============================
# 3. 小波变换：固定 db1（冻结）
# ============================

class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=2):
        super().__init__()
        self.register_buffer('dec_lo', torch.tensor([0.70710678, 0.70710678], dtype=torch.float32))
        self.register_buffer('dec_hi', torch.tensor([-0.70710678, 0.70710678], dtype=torch.float32))
        self.llproj = ConvBN(in_channels, out_channels, 1)
        self.lhproj = ConvBN(in_channels, out_channels, 1)
        self.hlproj = ConvBN(in_channels, out_channels, 1)
        self.hhproj = ConvBN(in_channels, out_channels, 1)

    def forward(self, x):
        B, C_in, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        lo, hi = self.dec_lo, self.dec_hi
        filters = torch.stack([
            lo[:, None] * lo[None, :],
            lo[:, None] * hi[None, :],
            hi[:, None] * lo[None, :],
            hi[:, None] * hi[None, :]
        ], dim=0)
        dec_filters = filters[:, None].repeat(C_in, 1, 1, 1).to(x.device)
        
        coeffs = F.conv2d(x, dec_filters, stride=2, groups=C_in, padding=0)
        H_out = (H + pad_h) // 2
        W_out = (W + pad_w) // 2
        coeffs = coeffs.view(B, C_in, 4, H_out, W_out)
        ll = self.llproj(coeffs[:, :, 0])
        lh = self.lhproj(coeffs[:, :, 1])
        hl = self.hlproj(coeffs[:, :, 2])
        hh = self.hhproj(coeffs[:, :, 3])
        return ll, lh, hl, hh


# ============================
# 4. Block 与 Backbone（无 LayerNorm）
# ============================

class WaveletDualAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, resolution=14):
        super().__init__()
        # High-frequency path
        self.high_pw = ConvBN(dim * 3, dim, 1)
        self.high_act = nn.GELU()
        self.high_attn = GlobalMHSAWithPosBias(dim, num_heads, resolution)
        self.high_dw = ConvBN(dim, dim, 3, 1, 1, dim)
        self.high_mlp = FFN(dim)

        # Low-frequency path
        self.low_ese = EffectiveSELayer(dim)
        self.low_attn = GlobalCrossAttnWithPosBias(dim, num_heads, resolution)
        self.low_dw = ConvBN(dim, dim, 3, 1, 1, dim)
        self.low_mlp = FFN(dim)

    def forward(self, ll, lh, hl, hh):
        # High branch
        high = torch.cat([lh, hl, hh], dim=1)
        high = self.high_act(self.high_pw(high))
        high_attn = high + self.high_attn(high)
        high_dw = high_attn + self.high_dw(high_attn)
        high_mlp = high_dw + self.high_mlp(high_dw)

        # Low branch
        ll = self.low_ese(ll)
        ll_attn = ll + self.low_attn( high_mlp, ll)
        ll_dw = ll_attn + self.low_dw(ll_attn)
        ll = ll_dw + self.low_mlp(ll_dw)
        return ll_dw


class WaveletCenterNet(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[128, 256, 384],
                 depth=[2, 3, 3],
                 num_heads=[4, 4, 4],
                 window_size=None,
                distillation=None):
        super().__init__()
        self.img_size = img_size
        self.stem = nn.Sequential(
            ConvBN(in_chans, embed_dim[0] // 4, 3, 2, 1, act_layer=nn.GELU()),
            ConvBN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, act_layer=nn.GELU()),
            ConvBN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, act_layer=nn.GELU())
        )

        # === 修正：精确计算每个 stage 的实际分辨率 ===
        def compute_stage_resolutions(img_size, num_stages=3):
            feat_size = img_size // 8  # stem downsample 8x
            resolutions = []
            for _ in range(num_stages):
                if feat_size % 2 != 0:
                    feat_size += 1  # pad to even
                feat_size //= 2
                resolutions.append(feat_size)
            return resolutions

        self.resolutions = compute_stage_resolutions(img_size, num_stages=3)
        # ==========================================

        self.stages = nn.ModuleList()
        for i in range(3):
            stage = nn.ModuleList()
            wavelet = LearnableWaveletTransform(
                in_channels=embed_dim[i] if i == 0 else embed_dim[i-1],
                out_channels=embed_dim[i]
            )
            stage.append(wavelet)
            for _ in range(depth[i]):
                block = WaveletDualAttentionBlock(
                    embed_dim[i], 
                    num_heads[i], 
                    resolution=self.resolutions[i]
                )
                stage.append(block)
            self.stages.append(stage)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        ll = x
        for stage in self.stages:
            wavelet, blocks = stage[0], stage[1:]
            ll, lh, hl, hh = wavelet(ll)
            for block in blocks:
                ll = block(ll, lh, hl, hh)
        return self.head(ll)


# ============================
# 5. 模型配置
# ============================

def create_config(img_size, embed_dim, depth, num_heads, window_size=7):
    return {
        'img_size': img_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'window_size': window_size,
    }

WaveletCenterNet_T0 = create_config(
    img_size=192,
    embed_dim=[64, 128, 192],
    depth=[1, 2, 3],
    num_heads=[2, 4, 4],
    window_size=6
)

WaveletCenterNet_T1 = create_config(
    img_size=192,
    embed_dim=[128, 144, 192],
    depth=[1, 2, 2],
    num_heads=[2, 3, 3],
    window_size=6
)

WaveletCenterNet_T2 = create_config(
    img_size=224,
    embed_dim=[128, 192, 224],
    depth=[1, 2, 3],
    num_heads=[4, 3, 2],
    window_size=7
)

WaveletCenterNet_T3 = create_config(
    img_size=224,
    embed_dim=[128, 240, 320],
    depth=[1, 2, 3],
    num_heads=[4, 3, 4],
    window_size=7
)

WaveletCenterNet_T4 = create_config(
    img_size=224,
    embed_dim=[176, 368, 448],
    depth=[1,2,2],
    num_heads=[4, 4, 8],
    window_size=7
)

WaveletCenterNet_T5 = create_config(
    img_size=224,
    embed_dim=[192, 384, 768],
    depth=[3, 5, 6],
    num_heads=[6, 6, 8],
    window_size=7
)

CONFIGS = {
    'WaveletCenterNet_T0': WaveletCenterNet_T0,
    'WaveletCenterNet_T1': WaveletCenterNet_T1,
    'WaveletCenterNet_T2': WaveletCenterNet_T2,
    'WaveletCenterNet_T3': WaveletCenterNet_T3,
    'WaveletCenterNet_T4': WaveletCenterNet_T4,
    'WaveletCenterNet_T5': WaveletCenterNet_T5,
}


def build_model(name='WaveletCenterNet_T2', num_classes=1000, window_size_override=None):
    cfg = CONFIGS[name].copy()
    img_size = cfg.pop('img_size')
    window_size = window_size_override or cfg.pop('window_size')
    return WaveletCenterNet(
        img_size=img_size,
        num_classes=num_classes,
        window_size=window_size,
        **cfg
    )


# ============================
# 6. FLOPs/Params 测试
# ============================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in CONFIGS:
        cfg = CONFIGS[name]
        model = build_model(name, num_classes=1000).to(device).eval()
        x = torch.randn(1, 3, cfg['img_size'], cfg['img_size']).to(device)
        flops = FlopCountAnalysis(model, x).total() / 1e6
        params = parameter_count(model)[''] / 1e6
        print(f"{name:<25} | {cfg['img_size']}x{cfg['img_size']:<5} | "
              f"WS={cfg['window_size']} | FLOPs={flops:6.2f}M | Params={params:6.2f}M")