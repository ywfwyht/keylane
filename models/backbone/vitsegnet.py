'''
Date: 2023-01-11 02:36:31
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-11 06:22:01
FilePath: /K-Lane/home/work_dir/work/keylane/models/backbone/vitsegnet.py
'''
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class VitSegNet(nn.Module):
    # img_size是图片大小
    # patch_size是patch的大小
    # in_channels是图片的通道数
    # embed是将patch vector映射到的维度
    # dropout一般给0 所以写不写其实一样
    # Heads：multi-head self-attention (MSA）中的heads数量；
    def __init__(self, image_size=144,
                    patch_size=8,
                    channels=64,
                    dim=512,
                    depth=1,
                    heads=16,
                    out_channels=1024,
                    expansion_factor=4,
                    dim_head=64,
                    dropout=0,
                    emb_dropout=0,
                    is_with_shared_mlp=True,
                    cfg=None):
        super().__init__()
        # img_height, img_width = image_size
        # patch_height, patch_width = patch_size
        assert image_size % patch_size == 0, 'Image dimensions \
            must be divisible by the patch size.'
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        patch_dim = channels * patch_size * patch_size
        # patch embedding
        #这里是对块进行编码，将patch_height*patch_width的大小输出维度变成隐层dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        # class token
        # 叠加（cat)一个dim维的cls_token到输入上（随机生成的，可叠加可不叠加）

        # positional embedding 给每一个token添加位置信息 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        mlp_dim = int(dim * expansion_factor)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        temp_h = int(image_size / patch_size)
        self.rearrage = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=temp_h, p1=patch_size, p2=patch_size)
        
        if is_with_shared_mlp:
            self.is_with_shared_mlp = True
            self.shared_mlp = nn.Conv2d(in_channels=int(dim/(patch_size**2)), out_channels=out_channels, kernel_size=1)
        else:
            self.is_with_shared_mlp = False
    
    def forward(self, x):
        print('in ', x.shape)
        x = self.to_patch_embedding(x)
        print('embed  ', x.shape)
        _, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        print('+pos_embed ', x.shape)

        x = self.dropout(x)
        x = self.transformer(x)
        print('trans ', x.shape)
        x = self.rearrage(x)
        print('ddddd ', x.shape)

        if self.is_with_shared_mlp:
            x = self.shared_mlp(x)
        
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0):
        super().__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mlp_block(x)

#attention的输入和输出维度相同[1,num_patches+1,128]-->[num_patches+1,128],其目的是赋予不同patch不同的权重；
#给予不同的注意力
# multi-head self-attention (MSA）
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        #获得三个维度相同的向量q,k,v,然后q,k相乘获得权重，乘以scale,再经过softmax之后，乘到v上
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


if __name__ == "__main__":
    v = VitSegNet()

    img = torch.randn(1, 64, 144, 144)

    preds = v(img)
    print(v)

    print(preds.shape)