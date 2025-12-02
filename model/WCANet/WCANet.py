from base import ConvBN, WaveletDualAttentionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model import MODEL
from fvcore.nn import FlopCountAnalysis, flop_count_table



class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        # 可学习小波滤波器参数
        self.dec_lo = nn.Parameter(torch.randn(filter_size) * 0.1)
        self.dec_hi = nn.Parameter(torch.randn(filter_size) * 0.1)
        self.rec_lo = nn.Parameter(torch.randn(filter_size) * 0.1)
        self.rec_hi = nn.Parameter(torch.randn(filter_size) * 0.1)

        # 1×1 卷积：用于通道对齐（因为小波本身不改变通道）
        self.proj = ConvBN(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def get_dec_filters(self):
        dec_filters = torch.stack([
            self.dec_lo.unsqueeze(0) * self.dec_lo.unsqueeze(1),
            self.dec_lo.unsqueeze(0) * self.dec_hi.unsqueeze(1),
            self.dec_hi.unsqueeze(0) * self.dec_lo.unsqueeze(1),
            self.dec_hi.unsqueeze(0) * self.dec_hi.unsqueeze(1)
        ], dim=0)  # [4, 1, K, K]
        return dec_filters[:, None].repeat(self.in_channels, 1, 1, 1)  # [4*C_in, 1, K, K]

    def forward(self, x):
        """
        Input:  [B, C_in, H, W]
        Output: ll, lh, hl, hh ∈ [B, C_out, H//2, W//2]
        """
        B, C_in, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Wavelet requires even resolution"

        # Apply learnable wavelet decomposition
        dec_filters = self.get_dec_filters().to(x.device)
        pad = self.filter_size // 2
        coeffs = F.conv2d(x, dec_filters, stride=2, groups=C_in)  # [B, 4*C_in, H//2, W//2]
        print(coeffs.shape)
        # Reshape to [B, C_in, 4, H//2, W//2]
        coeffs = coeffs.view(B, C_in, 4, H // 2, W // 2)

        # Project to C_out
        ll = self.proj(coeffs[:, :, 0, :, :])  # [B, C_out, H//2, W//2]
        lh = self.proj(coeffs[:, :, 1, :, :])
        hl = self.proj(coeffs[:, :, 2, :, :])
        hh = self.proj(coeffs[:, :, 3, :, :])

        return ll, lh, hl, hh



class WaveletCenterNet_64x(nn.Module):
    def __init__(self,
                 img_size=256,                  # must be divisible by 64
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[128, 256, 384],     # 3 stages
                 depth=[1,2,3],
                 num_heads=[4, 4, 4],
                 filter_size=2):
        super().__init__()
        assert img_size % 64 == 0, "img_size must be divisible by 64 (e.g., 256)"
        self.embed_dim = embed_dim

        # -----------------------
        # Stem: 3 × ConvBN + GELU, each stride=2 → total ↓8
        # -----------------------
        self.stem = nn.Sequential(
            ConvBN(in_chans, embed_dim[0] // 4, 3, stride=2, padding=1, act_layer=nn.GELU()),
            ConvBN(embed_dim[0] // 4, embed_dim[0] // 2, 3, stride=2, padding=1, act_layer=nn.GELU()),
            ConvBN(embed_dim[0] // 2, embed_dim[0], 3, stride=2, padding=1)
        )
        stem_res = img_size // 8  # e.g., 256 → 32

        # -----------------------
        # 3 Stages with Learnable Wavelet Downsample (each ↓2)
        # -----------------------
        self.stages = nn.ModuleList()
        current_res = stem_res  # e.g., 32

        for i in range(3):  # 3 stages
            stage = nn.ModuleList()

            # Learnable Wavelet Transform (for Stage 0, input is stem output)
            wavelet = LearnableWaveletTransform(
                in_channels=embed_dim[i] if i == 0 else embed_dim[i-1],
                out_channels=embed_dim[i],
                filter_size=filter_size
            )
            stage.append(wavelet)
            current_res //= 2  # after wavelet

            # Add blocks
            for j in range(depth[i]):
                block = WaveletDualAttentionBlock(
                    dim=embed_dim[i],
                    num_heads=num_heads[i],
                    resolution=current_res
                )
                stage.append(block)

            self.stages.append(stage)

        # -----------------------
        # Head
        # -----------------------
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
        # Stem: ↓8
        x = self.stem(x)  # [B, C0, H/8, W/8], e.g., [B, 128, 32, 32]

        ll = x
        lh = hl = hh = torch.zeros_like(ll)  # dummy for Stage 0 wavelet

        for i, stage in enumerate(self.stages):
            # First module is LearnableWaveletTransform
            wavelet = stage[0]
            blocks = stage[1:]

            ll, lh, hl, hh = wavelet(ll)  # ↓2, and get 4 subbands

            for block in blocks:
                ll = block(ll, lh, hl, hh)

        return self.head(ll)


#@MODEL.register_module
#运算量：282.732M, 参数量：4.023M
def WaveletCenterNet_64x(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None):
    model = WaveletCenterNet_64x(num_classes=num_classes,)
    return model





if __name__ == "__main__":

    model = WaveletCenterNet_64x()
    model.eval()
    model.to("cuda")
    # x = torch.randn(64, 3,224,224).to("cuda")
    flops = FlopCountAnalysis(model, torch.randn([1, 3, 256,256]).to("cuda"))
    print(f"FLOPs: {flops.total():,}")
