from functools import partial

import random
import math

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from timm.models.layers import Mlp

from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed
import util.statistics as statistics
import torch.nn.functional as F

import torch
import torch.nn as nn

from util.patch_embed import PatchEmbed

from functools import partial
import pywt


############################################
#    1) Wavelet Conv Section 
############################################

def create_learnable_wavelet_filter(wave, in_size, out_size, init_scale=1.0, dtype=torch.float32):
    """
    Initialize decomposition and reconstruction filters from a given pywt.Wavelet 
    and set them as learnable parameters (nn.Parameter).
    """
    w = pywt.Wavelet(wave)

    # Decomposition filters
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters_2d = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ], dim=0)
    dec_filters_2d = dec_filters_2d[:, None].repeat(in_size, 1, 1, 1)
    dec_filters_2d = dec_filters_2d * init_scale
    dec_filters_2d = nn.Parameter(dec_filters_2d, requires_grad=True)

    # Reconstruction filters
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters_2d = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
    ], dim=0)
    rec_filters_2d = rec_filters_2d[:, None].repeat(out_size, 1, 1, 1)
    rec_filters_2d = rec_filters_2d * init_scale
    rec_filters_2d = nn.Parameter(rec_filters_2d, requires_grad=True)

    return dec_filters_2d, rec_filters_2d


def wavelet_transform(x, filters):
    """
    Forward wavelet transform using depthwise conv2d.

    Args:
        x: (B, C, H, W), input feature map
        filters: shape=[4*C, 1, kH, kW], used for depthwise wavelet decomposition
    Returns:
        out: (B, C, 4, newH, newW), where each channel has been decomposed into LL, LH, HL, HH
    """
    b, c, h, w = x.shape
    # Compute padding for 2D convolution (if kH,kW are odd => pad=(kW//2, kH//2))
    pad = (filters.shape[-1] // 2, filters.shape[-2] // 2)

    # Use x as input, filters as weights
    # bias=None means no bias; stride=2 for downsampling; groups=c for depthwise
    out = F.conv2d(
        x,
        filters,
        bias=None,
        stride=2,
        groups=c,
        padding=pad
    )
    # out.shape => [b, 4*c, newH, newW]

    b_out, c_out, newH, newW = out.shape
    # c_out should be 4*c
    # reshape => [b, c, 4, newH, newW]
    out = out.view(b_out, c, 4, newH, newW)
    return out


def inverse_wavelet_transform(x, filters):
    """
    Inverse wavelet transform using depthwise conv_transpose2d.

    Args:
        x: shape [B, C, 4, hh, ww], where hh, ww are sizes from wavelet_transform
        filters: shape [4*C, 1, kH, kW], used for depthwise reconstruction
    Returns:
        out: shape [B, C, outH, outW], the reconstructed feature map
    """
    b, c, four, hh, ww = x.shape
    # c*4 must match filters out_channels (4*C)
    out = x.reshape(b, c*4, hh, ww)  # => [B, 4*C, hh, ww]

    # Same padding logic as wavelet_transform
    pad = (filters.shape[3] // 2, filters.shape[2] // 2)  # (padW, padH)

    # Inverse convolution: stride=2 for upsampling, groups=c
    out = F.conv_transpose2d(
        out,
        filters,
        stride=2,
        groups=c,
        padding=pad
    )
    # out.shape => [B, C, someH, someW]
    return out


class _ScaleModule(nn.Module):
    """
    A learnable scaling module to multiply input by a parameter.
    """
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    """
    Learnable Wavelet + multi-level decomposition + high-frequency Dropout.
    Requires in_channels == out_channels for depthwise operation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=3,
        wt_type='db1',
        init_scale=1.0,
        highfreq_dropout=0.1
    ):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d: in_channels must == out_channels (depthwise)."

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.highfreq_dropout = highfreq_dropout

        # Learnable decomposition/reconstruction filters
        dec_filters, rec_filters = create_learnable_wavelet_filter(
            wave=wt_type,
            in_size=in_channels,
            out_size=in_channels,
            init_scale=init_scale
        )
        self.wt_filter = dec_filters
        self.iwt_filter = rec_filters

        # Partial wavelet transform
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # Depthwise conv for the lowest-frequency sub-band
        self.base_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding='same', stride=1, dilation=1,
            groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)

        # For each wavelet level, apply an extra conv on (LF + HF) sub-bands
        # wavelet_convs: shape=(C*4)->(C*4) depthwise
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels*4, in_channels*4, kernel_size,
                padding='same', stride=1, dilation=1,
                groups=in_channels*4, bias=False
            )
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])

        # Optional stride
        if stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            def do_stride(x_in):
                return F.conv2d(
                    x_in,
                    self.stride_filter,
                    bias=None,
                    stride=stride,
                    groups=in_channels
                )
            self.do_stride = do_stride
        else:
            self.do_stride = None

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        Performs wt_levels times wavelet decomposition -> wavelet reconstruction,
        and returns a feature of the same resolution with extra conv & dropout on HF.
        """
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # Forward wavelet decomposition multi-level
        for lvl in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # Pad if needed to ensure even dimensions
            if (curr_shape[2] % 2 != 0) or (curr_shape[3] % 2 != 0):
                pad_h = curr_shape[2] % 2
                pad_w = curr_shape[3] % 2
                curr_x_ll = F.pad(curr_x_ll, (0, pad_w, 0, pad_h))

            # Wavelet transform => [B, C, 4, H/2, W/2]
            dec_out = self.wt_function(curr_x_ll)

            # Split LF (LL) => dec_out[:, :, 0, ...]
            # and HF => dec_out[:, :, 1:4, ...]
            lf = dec_out[:, :, 0, :, :]
            hf = dec_out[:, :, 1:4, :, :]

            # Reshape => [B, C*4, H/2, W/2]
            out_reshape = dec_out.view(dec_out.shape[0], dec_out.shape[1]*4, dec_out.shape[3], dec_out.shape[4])

            # Apply depthwise conv on (LF + HF)
            out_reshape = self.wavelet_convs[lvl](out_reshape)
            out_reshape = self.wavelet_scale[lvl](out_reshape)
            # Reshape back => [B, C, 4, H/2, W/2]
            out_reshape = out_reshape.view(*dec_out.shape)

            # Re-split LF, HF
            lf = out_reshape[:, :, 0, :, :]
            hf = out_reshape[:, :, 1:4, :, :]

            # Optional dropout on HF
            if self.highfreq_dropout > 0.0 and self.training:
                hf = F.dropout(hf, p=self.highfreq_dropout, training=True)

            # Concatenate them back
            dec_out = torch.cat([lf.unsqueeze(2), hf], dim=2)  # => [B, C, 4, H/2, W/2]

            # Store LF for next level; store HF as well
            curr_x_ll = dec_out[:, :, 0, :, :]
            x_ll_in_levels.append(curr_x_ll)
            x_h_in_levels.append(dec_out[:, :, 1:4, :, :])

        # Inverse wavelet transform multi-level
        next_x_ll = 0
        for lvl in reversed(range(self.wt_levels)):
            curr_x_ll = x_ll_in_levels[lvl]
            curr_x_h = x_h_in_levels[lvl]
            curr_shape = shapes_in_levels[lvl]

            # Combine
            curr_x_ll = curr_x_ll + next_x_ll
            cat_4 = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)  # [B, C, 4, h_half, w_half]
            recon = self.iwt_function(cat_4)

            # Slice back to original resolution
            Horig, Worig = curr_shape[2], curr_shape[3]
            recon = recon[:, :, :Horig, :Worig]

            next_x_ll = recon

        x_tag = next_x_ll

        # Base conv on input x (lowest freq)
        x_base = self.base_conv(x)  # [B, C, H, W]
        x_base = self.base_scale(x_base)

        out = x_base + x_tag

        # Optional stride
        if self.do_stride is not None:
            out = self.do_stride(out)

        return out


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """
    Depthwise-separable conv + WaveletConv mixing.
    - Uses WTConv2d for depthwise part (in_channels==out_channels)
    - Then uses a pointwise conv to mix channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()
        # WTConv2d requires in==out for the depthwise part
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)
        # Pointwise to expand in_channels => out_channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)   # [B, in_channels, H, W]
        x = self.pointwise(x)   # [B, out_channels, H, W]
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for a given dimension.
    Assumes x has shape [B, nHeads, N, dim], where dim == self.dim (must be even).
    
    Steps:
      1) Generate cos, sin with size [N, dim//2] => unsqueeze to [N,1,1,dim//2] for broadcasting
      2) Permute x => [N, B, nHeads, dim], reshape => [N, B, nHeads, dim//2, 2]
      3) Split even/odd dimensions (x_even, x_odd), apply RoPE formula
      4) Restore to [N, B, nHeads, dim], then permute back to [B, nHeads, N, dim]
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        # Dimension must be even
        assert dim % 2 == 0, "RoPE dimension must be even."
        self.dim = dim
        self.base = base

    def forward(self, x, seq_len):
        """
        x: [B, nHeads, N, dim], where dim == self.dim
        seq_len: N (sequence length)
        return: Same dimension [B, nHeads, N, dim], with rotation applied
        """
        B, H, N, D = x.shape
        assert D == self.dim, f"RoPEEmbedding: dim mismatch, expect {self.dim}, got {D}"
        assert N == seq_len, f"RoPEEmbedding: seq_len {seq_len} vs x.shape[2] {N} mismatch"

        # Generate cos/sin
        # pos: [N], inv_freq: [dim//2], => freqs: [N, dim//2]
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=x.device).float() / D))
        freqs = pos[:, None] * inv_freq[None, :]  # => [N, dim//2]
        cos_val = torch.cos(freqs).unsqueeze(1).unsqueeze(1)  # => [N,1,1,dim//2]
        sin_val = torch.sin(freqs).unsqueeze(1).unsqueeze(1)  # => [N,1,1,dim//2]

        # Permute => [N, B, H, D]
        x_reshape = x.permute(2, 0, 1, 3)  # => [N, B, H, D]
        # Reshape => [N, B, H, (dim//2), 2], split even/odd dimensions
        x_reshape = x_reshape.reshape(N, B, H, D // 2, 2)
        x_even = x_reshape[..., 0]
        x_odd  = x_reshape[..., 1]

        # RoPE formula:
        # x_even' = x_even * cos - x_odd * sin
        # x_odd'  = x_even * sin + x_odd * cos
        x_even_out = x_even * cos_val - x_odd * sin_val
        x_odd_out  = x_even * sin_val + x_odd * cos_val

        # Combine back
        x_out = torch.stack([x_even_out, x_odd_out], dim=-1)
        x_out = x_out.reshape(N, B, H, D)  # => [N, B, H, D]

        # Permute back => [B, H, N, D]
        x_out = x_out.permute(1, 2, 0, 3)
        return x_out


class RoPEAttention(nn.Module):
    """
    Multi-head attention with RoPE applied to Q/K.
    1) Take input x: [B, N, C], where N=seq_len, C=embed_dim
    2) Compute linear qkv => Q,K,V: [B, H, N, head_dim]
    3) Apply RoPE on the first rope_dim channels of Q,K
    4) Compute attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, rope_dim=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # rope_dim usually equals head_dim
        if rope_dim is None:
            rope_dim = self.head_dim
        self.rope_dim = rope_dim
        assert self.rope_dim % 2 == 0, "rope_dim must be even"

        # Separate rotation for Q and K
        self.rope_q = RotaryEmbedding(self.rope_dim)
        self.rope_k = RotaryEmbedding(self.rope_dim)

    def forward(self, x, mask=None):
        """
        x: [B, N, C], batch_size=B, seq_len=N, embed_dim=C
        mask: [B, N] or [B, N, N] (optional)
        """
        B, N, C = x.shape
        # QKV => [3, B, H, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # => [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # => each [B, H, N, head_dim]

        # Apply RoPE to Q,K
        if self.rope_dim > 0 and self.rope_dim <= self.head_dim:
            rope_dim = self.rope_dim
            # Split
            q_rope, q_rem = q[..., :rope_dim], q[..., rope_dim:]
            k_rope, k_rem = k[..., :rope_dim], k[..., rope_dim:]

            # Rotate
            q_rope = self.rope_q(q_rope, seq_len=N)  # => [B, H, N, rope_dim]
            k_rope = self.rope_k(k_rope, seq_len=N)

            # Merge back
            q = torch.cat([q_rope, q_rem], dim=-1)  # => [B,H,N,head_dim]
            k = torch.cat([k_rope, k_rem], dim=-1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # => [B, H, N, N]

        if mask is not None:
            # If mask shape is [B, N, N] => broadcast to [B, H, N, N]
            # Can also be [B, 1, N, N] or [1, 1, N, N] etc.
            attn = attn + mask

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Get output => [B, H, N, head_dim]
        x_out = attn @ v

        # Reshape => [B, N, C]
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_map = None

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape # C = embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3) # (QKV, B, Heads, N, head_dim)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, Heads, N, head_dim)

        if attn_mask is not None:
            attn_mask = 1 - attn_mask
        attn, attn_weights = self.mha(q, k, v, key_padding_mask=attn_mask)
        self.attn_map = attn_weights

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x
    

class WaveFormer(nn.Module):
    """ 
    Open model for general time series analysis 
    """
    def __init__(self, domains: dict, domain_weights: dict, domain_agnostic: str = False, 
                 input_channels=1, time_steps=2500, patch_size=(1, 100),
                 embed_dim=1024, depth=24, num_heads=16,
                 output_projection='decoder',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, separate_dec_pos_embed_y=False,
                 head_mlp_ratio=4., head_dropout=0.1, head_activation=nn.GELU,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, masked_patch_loss=False, domain_weighted_loss=False, contrastive_loss=False,
                 include_forecasting=False, forecasting_probability=0.33, forecasting_mask_ratio=0.5,
                 downstream=None, args=None):
        """
        Args:
            domains: a dict describing domain-specific input shapes, e.g., {domain_name: (C, T)}
            domain_weights: a dict of domain-specific weights for domain-weighted training
            domain_agnostic: whether pos_embed_y is shared (domain-agnostic) or domain-specific
            input_channels: number of channels in the input
            time_steps: total time length
            patch_size: (patch_height, patch_width)
            embed_dim: embedding dimension for the Transformer
            depth: number of Transformer blocks
            num_heads: number of attention heads
            output_projection: 'decoder' or 'mlp' output type
            decoder_embed_dim: dimension for the decoder embedding
            decoder_depth: number of Transformer blocks in the decoder
            decoder_num_heads: number of attention heads in the decoder
            separate_dec_pos_embed_y: whether y-axis positional embedding is separated
            head_mlp_ratio: expansion ratio for the MLP in the final head
            head_dropout: dropout rate for the head MLP
            head_activation: activation function for the head MLP
            mlp_ratio: expansion ratio for the main Transformer MLP blocks
            norm_layer: normalization layer to use
            norm_pix_loss: whether to normalize pixel-level reconstruction losses
            masked_patch_loss: whether masked patch loss is active
            domain_weighted_loss: whether to weight losses by domain
            contrastive_loss: whether to include contrastive loss
            include_forecasting: whether to include forecasting steps
            forecasting_probability: probability of choosing a forecasting step
            forecasting_mask_ratio: ratio of masked positions for forecasting
            downstream: downstream task type, e.g., 'classification' or 'regression'
            args: additional arguments, typically includes nb_classes for classification/regression
        """
        super().__init__()

        # Encoder specifics
        self.patch_size = patch_size
        # Patch embedding (Conv2d with kernel_size=stride=patch_size)
        self.patch_embed = PatchEmbed(input_channels, patch_size, embed_dim, flatten=False)
        # If you used FreqPatchEmbed, you can uncomment the following line:
        # self.patch_embed = FreqPatchEmbed(in_chans=input_channels, patch_size=patch_size,
        #                                   embed_dim=embed_dim, n_fft=64, hop_length=32, flatten=False)

        # Wavelet convolution module
        # Because WTConv2d requires in_channels == out_channels, both are set to embed_dim
        self.wavelet_conv = DepthwiseSeparableConvWithWTConv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,  # must match in_channels for WTConv2d
            kernel_size=3
        ) 

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Norm for the final CLS feature before the head
        self.fc_norm = norm_layer(embed_dim)

        # Domain-specific or domain-agnostic vertical grid info
        self.grid_height = {}
        for domain, input_size in domains.items():
            # Number of variates = input_size[1] // patch_size[0]
            grid_height = input_size[1] // patch_size[0]
            self.grid_height.update({domain: grid_height})

        # Horizontal pos_embed dimension
        assert embed_dim % 2 == 0
        max_num_patches_x = time_steps // patch_size[1]
        self.max_num_patches_x = max_num_patches_x
        # +1 for the cls token position
        self.pos_embed_x = nn.Parameter(torch.zeros(1, max_num_patches_x + 1, embed_dim // 2), requires_grad=False)

        # Domain-agnostic or domain-specific pos_embed_y
        self.domain_agnostic = domain_agnostic
        if self.domain_agnostic:
            # Shared pos_embed_y
            total_num_embeddings_y = 256
        else:
            # Domain-specific pos_embed_y
            total_num_embeddings_y = sum([v for k, v in self.grid_height.items()])

        self.pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, embed_dim // 2, padding_idx=0)  # +1 for padding

        # Main Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Modify the attention operation to handle attention masks if needed
        for block in self.blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn = RoPEAttention(
                dim=embed_dim,       # Or decoder_embed_dim if using in decoder
                num_heads=num_heads,
                qkv_bias=True,
                rope_dim=None,       # Default uses head_dim, can be customized
                attn_drop=0.0,
                proj_drop=0.0
            )

        # Output projection specifics
        self.output_projection = output_projection

        if self.output_projection == 'mlp':
            # Mask token encoder for masked patch tasks
            self.mask_token_encoder = nn.Parameter(torch.zeros(1, 1, embed_dim))

            # MLP for final projection
            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * head_mlp_ratio),
                act_layer=head_activation,
                drop=head_dropout,
            )
            self.mlp_norm = norm_layer(embed_dim)
            self.mlp_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True)

        else:  # 'decoder' branch
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            assert decoder_embed_dim % 2 == 0

            self.decoder_pos_embed_x = nn.Parameter(
                torch.zeros(1, max_num_patches_x + 1, decoder_embed_dim // 2), requires_grad=False
            )
            self.separate_dec_pos_embed_y = separate_dec_pos_embed_y
            if self.separate_dec_pos_embed_y:
                self.decoder_pos_embed_y = nn.Embedding(total_num_embeddings_y + 1, decoder_embed_dim // 2,
                                                        padding_idx=0)
            else:
                self.decoder_pos_embed_y = nn.Linear(embed_dim // 2, decoder_embed_dim // 2)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                      act_layer=nn.GELU, norm_layer=norm_layer)
                for i in range(decoder_depth)
            ])
            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim,
                                          patch_size[0] * patch_size[1] * input_channels,
                                          bias=True)
            # Likewise, modify these blocks to handle attn masks if needed
            for block in self.decoder_blocks:
                block.forward = self._block_forward_wrapper(block)
                block.attn = Attention(decoder_embed_dim, decoder_num_heads, qkv_bias=True)

        # Contrastive specifics
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        proj_dim = int(1024)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False)
        )

        pred_dim = int(128)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, embed_dim, bias=False),
        )

        # Misc. settings
        self.norm_pix_loss = norm_pix_loss
        self.masked_patch_loss = masked_patch_loss

        self.domain_weights = domain_weights
        self.domain_weighted_loss = domain_weighted_loss

        self.contrastive_loss = contrastive_loss

        self.include_forecasting = include_forecasting
        self.forecasting_probability = forecasting_probability
        self.forecasting_mask_ratio = forecasting_mask_ratio

        self.downstream = downstream
        # Define head + fc_norm according to downstream tasks
        if self.downstream == 'classification':
            # Classification requires args.nb_classes
            self.head = nn.Linear(embed_dim, args.nb_classes)
            self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        elif self.downstream == 'regression':
            # Regression tasks => self.head for output dimension
            self.head = nn.Linear(embed_dim, args.nb_classes)
            self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        else:
            self.head = nn.Identity()
            self.fc_norm = nn.Identity()

        self.initialize_weights()

    def activate_masked_loss(self):
        """Enable masked patch loss."""
        self.masked_patch_loss = True

    def _block_forward_wrapper(self, block_obj):
        """
        A customized forward for the timm.models.vision_transformer Block
        that can optionally handle attention masks.
        """
        def my_forward(x, attn_mask=None):
            x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask)))
            x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
            return x
        return my_forward

    def initialize_weights(self):
        """Initialize all learnable parameters including positional embeddings, tokens, and heads."""
        # Initialize learnable pos_embed for the vertical axis
        _pos_embed_y = torch.nn.Parameter(
            torch.randn(self.pos_embed_y.num_embeddings - 1, self.pos_embed_y.embedding_dim) * 0.02
        )
        trunc_normal_(_pos_embed_y, std=0.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_embed_y

        # If separate decoder pos_embed_y is used
        if self.output_projection == "decoder" and self.separate_dec_pos_embed_y:
            _decoder_pos_embed_y = torch.nn.Parameter(
                torch.randn(self.decoder_pos_embed_y.num_embeddings - 1, self.decoder_pos_embed_y.embedding_dim) * 0.02
            )
            trunc_normal_(_decoder_pos_embed_y, std=0.02)
            with torch.no_grad():
                self.decoder_pos_embed_y.weight[1:] = _decoder_pos_embed_y

        # Initialize (and freeze) pos_embed_x with sin-cos embedding
        _pos_embed_x = get_1d_sincos_pos_embed(
            self.pos_embed_x.shape[-1],
            self.pos_embed_x.shape[-2] - 1,
            cls_token=True
        )
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_embed_x).float().unsqueeze(0))

        if self.output_projection == "decoder":
            _decoder_pos_embed_x = get_1d_sincos_pos_embed(
                self.decoder_pos_embed_x.shape[-1],
                self.decoder_pos_embed_x.shape[-2] - 1,
                cls_token=True
            )
            self.decoder_pos_embed_x.data.copy_(torch.from_numpy(_decoder_pos_embed_x).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # If decoder, init mask_token
        if self.output_projection == "decoder":
            torch.nn.init.normal_(self.mask_token, std=0.02)
        else:  # mlp
            torch.nn.init.normal_(self.mask_token_encoder, std=0.02)

        # Initialize nn.Linear and nn.LayerNorm layers
        self.apply(self._init_weights)

        # Initialize the final classification/regression head if needed
        if hasattr(self, 'head') and isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.01)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def _init_weights(self, m):
        """
        A helper function to initialize linear and LayerNorm layers with xavier/uniform.
        """
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following the official JAX ViT approach
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert a (N, C, H, W) tensor into patch embeddings: (N, L, p*q*C)

        Args:
            imgs: shape (N, C, H, W)
        Returns:
            x: shape (N, L, p*q*C), where L = (H/p) * (W/q)
        """
        p, q = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % q == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // q
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * q * imgs.shape[1]))
        return x

    def forward_encoder_all_patches(self, x):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            pos_embed_y: (B, C', T'), with N=C'*T' embedding ids

        output:
            x: (B, 1+N, D), with 1 cls token + N (visible + masked) patches
        """
        # Embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        # Wavelet: [B, embed_dim, H', W']
        # x = self.wavelet_conv(x)

        # Flatten => [B, N, D]
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp*Wp).transpose(1, 2)  # => (B, N, C)

        # CLS token:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # => [B, 1, D]
        x = torch.cat((cls_token, x), dim=1)  # => [B, 1+N, D]

        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)  # no attn_mask

        # Norm
        x = self.norm(x)

        return x

    def forward_features(self, x, pos_embed_y):
        """
        Extract embedding features from the input x and pos_embed_y, excluding the final prediction head.
        It is assumed that in the downstream finetuning process, this method will be called to obtain 
        the embedding, followed by calling forward_head for the final prediction.
        """
        # Use forward_encoder_all_patches to extract features from all patches without masking,
        # to obtain a stable embedding.
        # If masking is needed, modify this part accordingly.

        x = self.forward_encoder_all_patches(x)  # x: (B, 1+N, D)

        # Take the CLS token as the global feature (B, D)
        emb = x[:, 0]

        return emb

    def forward_head(self, x, pre_logits: bool = False):
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, pos_embed_y):
        x = self.forward_features(x, pos_embed_y)
        x = self.forward_head(x)
        return x


def Waveformer_base(**kwargs):
    model = WaveFormer(
        embed_dim=256, depth=6, num_heads=8,                               # dim=64 per head
        decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=1,        # dim=32 per head
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
