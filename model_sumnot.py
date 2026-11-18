## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torchvision.utils import save_image
from einops import rearrange
from einops.layers.torch import Rearrange
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


def build_mlp(hidden_size, projector_dim, z_dim):
    """
    Builds the MLP projector.
    Args:
        hidden_size: Input dimension (Channel count of Restormer bottleneck).
        projector_dim: Hidden dimension.
        z_dim: Output dimension (Aligned with DINO feature dim, e.g., 384).
    """
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# ==================================================================
# 2. Restormer Main Class (Integrated with MLP)
# ==================================================================
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 decoder=False,
                 # [New Arguments]
                 use_projector=True,      # Toggle internal MLP projector
                 projector_out_dim=384    # Output dimension
                 ):

        super(Restormer, self).__init__()

        self.use_projector = use_projector
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.decoder = decoder

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        
        # -------------------------------------------------------
        # Level 1
        # -------------------------------------------------------
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        
        # -------------------------------------------------------
        # Level 2
        # -------------------------------------------------------
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        # ============================================================
        # [修改点 1] 初始化 Projector (针对 Level 2)
        # Level 2 通道数 = dim * 2^1
        # ============================================================
        self.projector_out_dim = projector_out_dim
        if self.use_projector:
            level2_dim = int(dim * 2 ** 1)
            self.projector = nn.Conv2d(
                in_channels=level2_dim, 
                out_channels=self.projector_out_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=True
            )
        # ============================================================

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        
        # -------------------------------------------------------
        # Level 3
        # -------------------------------------------------------
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        # -------------------------------------------------------
        # Latent (Level 4)
        # -------------------------------------------------------
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.latent_dim = int(dim * 2 ** 3)

        self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        
        self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 192, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)

        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        
        self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1

        self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # ============================================================
        # [修改点 2] 在 Level 2 输出后直接计算 Projection
        # ============================================================
        projected_feat = None
        if self.use_projector:
            # 1. 投影
            x = self.projector(out_enc_level2) 

            # 2. 计算统计量
            u = x.mean(1, keepdim=True)
            s = x.std(1, keepdim=True)

            # 3. 归一化 (x - 0.7*u) / (s + eps)
            x_norm = (x - 0.7 * u) / (s + 1e-6)

            projected_feat = x_norm
        # ============================================================

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, projected_feat
    
# class Restormer(nn.Module):
#     def __init__(self,
#                  inp_channels=3,
#                  out_channels=3,
#                  dim=48,
#                  num_blocks=[4, 6, 6, 8],
#                  num_refinement_blocks=4,
#                  heads=[1, 2, 4, 8],
#                  ffn_expansion_factor=2.66,
#                  bias=False,
#                  LayerNorm_type='WithBias',  ## Other option 'BiasFree'
#                  decoder=False,
#                  # [New Arguments]
#                  use_projector=True,      # Toggle internal MLP projector
#                  projector_out_dim=384    # Output dimension (z_dim) to align with DINO/Teacher
#                  ):

#         super(Restormer, self).__init__()

#         self.use_projector = use_projector
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

#         self.decoder = decoder

#         self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
#         self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
#         self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

#         self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
#         self.encoder_level1 = nn.Sequential(*[
#             TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                              LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

#         self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

#         self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
#         self.encoder_level2 = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

#         self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

#         self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
#         self.encoder_level3 = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

#         self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

#         # Latent Layer Definition
#         self.latent = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
#         # [Crucial] Calculate Latent Channel Dimension
#         # Level 4 dimension is: dim * 2^3 = dim * 8
#         self.latent_dim = int(dim * 2 ** 3)
#         self.projector_out_dim = projector_out_dim
#         # ============================================================
#         # [New] Initialize MLP Projector
#         # ============================================================
#         if self.use_projector:
#             # Typically, hidden dim is 2x input dim
#            self.projector = nn.Conv2d(
#     in_channels=self.latent_dim, 
#     out_channels=self.projector_out_dim, 
#     kernel_size=3, 
#     stride=1, 
#     padding=1,      # 【关键】必须是1，保证输出 H,W 不变
#     bias=True
# )

#         self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        
#         self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 192, num_heads=heads[2],
#                                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                                              LayerNorm_type=LayerNorm_type)

#         self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)

#         self.decoder_level3 = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

#         self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        
#         self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) * 2, num_heads=heads[2],
#                                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                                              LayerNorm_type=LayerNorm_type)
#         self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)

#         self.decoder_level2 = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

#         self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1

#         self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2],
#                                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                                              LayerNorm_type=LayerNorm_type)
#         self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)

#         self.decoder_level1 = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

#         self.refinement = nn.Sequential(*[
#             TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

#         self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

#     def forward(self, inp_img, noise_emb=None):

#         inp_enc_level1 = self.patch_embed(inp_img)

#         out_enc_level1 = self.encoder_level1(inp_enc_level1)

#         inp_enc_level2 = self.down1_2(out_enc_level1)

#         out_enc_level2 = self.encoder_level2(inp_enc_level2)

#         inp_enc_level3 = self.down2_3(out_enc_level2)

#         out_enc_level3 = self.encoder_level3(inp_enc_level3)

#         inp_enc_level4 = self.down3_4(out_enc_level3)
#         latent = self.latent(inp_enc_level4)

#         # This latent is what we want to align
#         latent_task = latent

#         # ============================================================
#         # [New] MLP Projection Logic
#         # ============================================================
#         projected_feat = None
#         if self.use_projector:
#             # nn.Linear works on the last dimension, so we need to permute.
#             # latent_task shape: [B, C, H, W]  
#             x = self.projector(latent_task) 


#             u = x.mean(1, keepdim=True)
#             s = x.std(1, keepdim=True)

#             # 3. 归一化 (加上了你之前的 0.7 系数，如果不需要改成 1.0 即可)
#             # 核心修正：这里除以了 (s + eps)
#             x_norm = (x - 0.7 * u) / (s + 1e-6)

#             projected_feat = x_norm

#         # mask = nn.Parameter(torch.randn_like(latent), requires_grad=True)
#         if self.decoder:
#             latent = self.noise_level3(latent)
#             latent = self.reduce_noise_level3(latent)

#         inp_dec_level3 = self.up4_3(latent)

#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

#         out_dec_level3 = self.decoder_level3(inp_dec_level3)
#         if self.decoder:
#             out_dec_level3 = self.noise_level2(out_dec_level3)
#             out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

#         inp_dec_level2 = self.up3_2(out_dec_level3)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

#         out_dec_level2 = self.decoder_level2(inp_dec_level2)
#         if self.decoder:
#             out_dec_level2 = self.noise_level1(out_dec_level2)
#             out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

#         inp_dec_level1 = self.up2_1(out_dec_level2)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

#         out_dec_level1 = self.decoder_level1(inp_dec_level1)

#         out_dec_level1 = self.refinement(out_dec_level1)

#         out_dec_level1 = self.output(out_dec_level1) + inp_img

#         # Return: (Restored Image, Projected Feature)
#         return out_dec_level1, projected_feat

class Transporter(nn.Module):
    def __init__(self,
                 dim=48,
                 num_blocks=4,
                 heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):

        super(Transporter, self).__init__()

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])


    def forward(self, latent):

        latent_projected = self.latent(latent)

        return latent_projected


class Potentials(nn.Module):
    def __init__(self, num_potentials, channels=384, size=128):
        super(Potentials, self).__init__()
        self.num_potentials = num_potentials
        self.input_channels = channels
        # self.num_features = int(size * size / 2)
        self.num_features = int(channels * (size / 32) * (size / 32))
        self.potentials = nn.ModuleList(
            [self._build_potentials(self.input_channels, self.num_features) for _ in range(num_potentials)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def _build_potentials(self, input_channels, num_fea):
        return nn.Sequential(

            # # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=4,groups=4, padding=1,
                      bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Flatten(),
            nn.Linear(num_fea, int(num_fea / 4)),
            nn.Linear(int(num_fea / 4), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, input, idx=None):
        # 删掉 input = input.unsqueeze(0) ！！！
        if idx is not None:
            return self.potentials[idx](input)
        else:
            return [potential(input) for potential in self.potentials]

class Potentials_shared(nn.Module):
    def __init__(self, num_potentials, channels=384, size=128, *, num_sources):
        super().__init__()
        if num_potentials != 2:
            raise ValueError("Expected two heads: conditioned source and HQ.")
        if not isinstance(num_sources, int) or num_sources < 1:
            raise ValueError("num_sources must be a positive integer.")
        if channels < 4 or channels % 4:
            raise ValueError("channels must be positive and divisible by 4.")
        self.num_potentials = num_potentials
        self.num_sources = num_sources
        self.input_channels = channels
        # Preserve the original size convention; this is not the input H/W.
        self.num_features = int(channels * (size / 32) * (size / 32))
        if self.num_features < 4:
            raise ValueError("size is too small for the original MLP.")
        self.potentials = nn.ModuleList([
            self._build_potentials(channels, self.num_features, num_sources),
            self._build_potentials(channels, self.num_features, 0),
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _build_potentials(self, input_channels, num_fea, condition_dim):
        return nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=4,
                      stride=4, groups=4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(num_fea + condition_dim, num_fea // 4),
            nn.Linear(num_fea // 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def _source(self, features, condition):
        if condition is None:
            raise ValueError("Source potential requires a domain condition.")
        if isinstance(condition, int):
            if not 0 <= condition < self.num_sources:
                raise ValueError("Domain condition is outside [0, num_sources).")
            labels = torch.full((features.shape[0],), condition,
                                device=features.device, dtype=torch.long)
        elif torch.is_tensor(condition):
            if condition.dtype not in (torch.int32, torch.int64):
                raise TypeError("Domain conditions must be integer IDs.")
            labels = condition.to(device=features.device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError("Expected one domain ID per feature sample.")
        else:
            raise TypeError("condition must be an integer or integer tensor.")

        # one_hot rejects negative IDs and IDs >= num_sources.
        code = F.one_hot(labels, num_classes=self.num_sources)
        branch = self.potentials[0]
        hidden = features
        for layer in list(branch.children())[:3]:
            hidden = layer(hidden)
        if hidden.shape[1] != self.num_features:
            raise ValueError(
                f"Flattened convolution has {hidden.shape[1]} features; "
                f"expected {self.num_features}. Check the size argument."
            )
        hidden = torch.cat((hidden, code.to(dtype=hidden.dtype)), dim=1)
        for layer in list(branch.children())[3:]:
            hidden = layer(hidden)
        return hidden

    def forward(self, input, idx=None, condition=None):
        if input.ndim != 4 or input.shape[1] != self.input_channels:
            raise ValueError("Expected features shaped [B, channels, H, W].")
        if idx == 0:
            return self._source(input, condition)
        if idx == 1:
            return self.potentials[1](input)
        if idx is not None:
            raise ValueError("idx must be 0 (source), 1 (HQ), or None.")
        return [self._source(input, condition), self.potentials[1](input)]


def shared_potential_values(pots, sources, hq):
    """Evaluate f(z_k, k), k=0,...,K-1, and unconditional f_HQ(z).

    Domain index k denotes the actual source distribution, not a shuffled
    sample slot. The caller is responsible for correct per-domain sampling.
    """
    module = pots.module if hasattr(pots, "module") else pots
    if len(sources) != module.num_sources:
        raise ValueError("Provide one feature batch per configured source domain.")
    if hq.ndim != 4 or hq.shape[0] == 0:
        raise ValueError("HQ features must have nonempty shape [N, C, H, W].")
    if any(source.shape != hq.shape for source in sources):
        raise ValueError("Every source and HQ batch must have identical shapes.")
    n = hq.shape[0]
    z_all = torch.cat([*sources, hq], dim=0)
    conditions = torch.cat([
        torch.full((n,), k, device=hq.device, dtype=torch.long)
        for k in range(len(sources))
    ] + [torch.zeros(n, device=hq.device, dtype=torch.long)])
    # The source-head output for HQ rows uses dummy condition 0 and is unused.
    # Similarly, HQ-head outputs for source rows are unused.
    values = pots(z_all, condition=conditions)
    f_sources = [values[0][k*n:(k+1)*n].reshape(-1)
                 for k in range(len(sources))]
    f_hq = values[1][len(sources)*n:].reshape(-1)
    return f_sources, f_hq


