'''
Date: 2023-01-11 03:08:55
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-01-18 03:10:14
FilePath: /K-Lane/home/work_dir/work/keylane/models/neck/datnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import trunc_normal_, to_2tuple

class DatNet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=10, expansion=4,
                dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                heads=[3, 6, 12, 24], window_sizes=[7, 7, 7, 7],
                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                strides=[-1, -1, -1, -1], offset_range_factor=[1, 2, 3, 4], 
                stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'],
                ['L', 'D']], groups=[-1, -1, 3, 6], use_pes=[False, False, False, False],
                dwc_pes=[False, False, False, False], sr_ratios=[8, 4, 2, 1],
                fixed_pes=[False, False, False, False], no_offs=[False, False, False, False],
                ns_per_pts=[4, 4, 4, 4], use_dwc_mlps=[False, False, False, False],
                use_conv_patches=False, **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i], dim1, dim2,
                depths[i], stage_spec[i], groups[i], use_pes[i], sr_ratios[i], heads[i],
                strides[i], offset_range_factor[i], i, dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, dpr[sum(depths[:i]):sum(depths[:i+1])],
                use_dwc_mlps[i])
            )
            img_size = img_size // 2
        self.down_projs = nn.ModuleList()

        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i+1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i+1])
                )
            )
        self.cls_norm = LayerNormProxy(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.date
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.date
                if 'reference' in keys:
                    new_state_dict[state_key] = m.date
                # Bicubic Interpolation
                if 'relative_positive_bias_table' in keys:
                    n, c = state_value.size()
                    import math
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    c, H, W = m.date.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def forward(self, x):
        x = self.patch_proj(x)
        positions = []
        references = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)

        return x, positions, references


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt, dim_in, dim_embed, depths,
                stage_spec, n_groups, use_pe, sr_ratio, heads, stride, offset_range_factor,
                stage_idx, dwc_pe, no_off, fixed_pe, attn_drop, proj_drop, expansion,
                drop, drop_path_rate, use_dwc_mlp):
        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)
                if use_dwc_mlp else TransformerMLPWithLinear(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )

        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop))
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DeformAttention(fmap_size, fmap_size, heads, hc, n_groups, attn_drop,
                    proj_drop, stride, offset_range_factor, use_pe, dwc_pe, no_off,
                    fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                import math
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop,
                    shift_size, fmap_size,)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class LocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.proj_qkv = nn.Linear(dim, 3*dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2 Wh Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relativate_coords = coords_flatten[:,:, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relativate_coords = relativate_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relativate_coords[:, :, 0] += self.window_size[0] - 1 # shift to start from 0
        relativate_coords[:, :, 1] += self.window_size[1] - 1
        relativate_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relativate_position_index = relativate_coords.sum(-1) # Wh*Ww, Wh*Ww
        self.register_buffer("relation_position_index", relativate_position_index)

    def forward(self, x, mask=None):
        B,C,H,W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c',
            h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')
        qkv = self.proj_qkv(x_total) # B' N 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q,k,v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn: (b * nW) h w w
            # mask: nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, 
                w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))
        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x)) # B' N C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2,
            h1=self.window_size[0], w1=self.window_size[1]) # B C H W
        
        return x, None, None

class ShiftWindowAttention(LocalAttention):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)
        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size
        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size) # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)',
            h1=self.window_size[0], w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None

class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2),
            nn.GELU(),
            nn.Dropout(drop, inplace=True),
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            nn.Dropout(drop, inplace=True)
        )
    
    def forward(self, x):

        x = self.mlp(x)
        
        return x

class TransformerMLPWithLinear(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.mlp_chunk = nn.Sequential(
            nn.Linear(self.dim1, self.dim2),
            nn.GELU(),
            nn.Dropout(drop, inplace=True),
            nn.Linear(self.dim2, self.dim1),
            nn.Dropout(drop, inplace=True)
        )
  
    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.mlp_chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class DeformAttention(nn.Module):
    def __init__(self, q_size, kv_size, heads, head_channels, groups,
                    attn_drop, proj_drop, stride, offset_range_factor,
                    use_pe, dwc_pe, no_off, fixed_pe, stage_idx):
        super().__init__()
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.heads = heads
        self.head_channels = head_channels
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.nc = head_channels * heads
        self.group_channels = self.nc // self.groups
        self.group_heads = self.heads // self.groups
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, kk, stride, kk//2, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv2d(self.group_channels, 2, 1, 1, 0, bias=False)
        )
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3,
                                        stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.heads, self.kv_h * 2 -1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
    
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key-0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key-0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1) # B * g H W 2

        return ref
    
    def forward(self, x):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w',
                                g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1. / Hk, 1. / Wk], device=device).reshape(1,2,1,1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.)
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        x_sample = F.grid_sample(
            input=x.reshape(B * self.groups, self.group_channels, H, W),
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True
        )
        x_sample = x_sample.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.heads, self.head_channels, H * W)
        k = self.proj_k(x_sample).reshape(B * self.heads, self.head_channels, n_sample)
        v = self.proj_v(x_sample).reshape(B * self.heads, self.head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.heads,
                    self.head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.heads, H * W, n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.groups, H*W, 2).unsqueeze(2) - 
                    pos.reshape(B * self.groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.groups, self.group_heads, 2 * H - 1, 2 * W -1),
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                attn_bias = attn_bias.reshape(B * self.heads, H * W, n_sample)
                attn = attn + attn_bias
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.groups, Hk, Wk, 2), reference.reshape(B, self.groups, Hk, Wk, 2)



class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)

        return einops.rearrange(x, 'b h w c -> b c h w')

if __name__=="__main__":

    dat = DatNet(img_size=224,
            patch_size=4,
            num_classes=1000,
            expansion=4,
            dim_stem=128,
            dims=[128, 256, 512, 1024],
            depths=[2, 2, 18, 2],
            stage_spec=[['L', 'S'], ['L', 'S'], 
            ['L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D', 'L', 'D'], 
            ['L', 'D']],
            heads=[4, 8, 16, 32],
            window_sizes=[7, 7, 7, 7],
            groups=[-1, -1, 4, 8],
            use_pes=[False, False, True, True],
            dwc_pes=[False, False, False, False],
            strides=[-1, -1, 1, 1],
            sr_ratios=[-1, -1, -1, -1],
            offset_range_factor=[-1, -1, 2, 2],
            no_offs=[False, False, False, False],
            fixed_pes=[False, False, False, False],
            use_dwc_mlps=[False, False, False, False],
            use_conv_patches=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.5)

    print(dat)