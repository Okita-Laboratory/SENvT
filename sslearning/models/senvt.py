import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PatchEmbed(nn.Module):
    def __init__(self, dim=1024, window_size=300, patch_size=1, in_chans=3, is_eva=False):
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_patches = window_size // patch_size
        self.is_eva = is_eva
        
        self.proj = nn.Conv1d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.mask_proj = nn.Conv1d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-5, bias=True)
    def mask_patchfy(self, m):
        m = self.mask_proj(m.unsqueeze(1)).squeeze(1).to(dtype=torch.bool)
        return m
    def forward(self, x, mask=None):
        B, C, W = x.shape
        assert self.window_size == W, 'Error: window size not correct.'
        
        if self.is_eva or mask is None:
            x = self.proj(x).flatten(-1).transpose(1, 2)
            x = self.norm(x)
            return x, None
        else:
            mask = self.mask_patchfy(mask)
            x = self.proj(x)
            x = x.flatten(-1).transpose(1, 2) # BCW -> BWC
            x = self.norm(x)
            return x, mask


class Attention(nn.Module):
    def __init__(self, dim, nheads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.nheads = nheads
        head_dim = dim // nheads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def mask_(self, mask):
        N, L = mask.shape
        cls_mask = torch.zeros(N, 1, dtype=torch.bool, device=mask.device) # N, 1
        mask = torch.cat([cls_mask, mask], dim=1)                          # N, L+1
        mask = mask.unsqueeze(1).repeat(1, L+1, 1)                         # N, L+1, L+1
        mask = mask.unsqueeze(1).repeat(1, self.nheads, 1, 1)              # N, nheads, L+1, L+1
        return mask
    
    def forward(self, x, mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nheads, C//self.nheads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = self.mask_(mask)
            attn = attn.masked_fill(mask, torch.tensor(float('-inf'), dtype=x.dtype))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MLP(nn.Module):
    def __init__(self, in_dim, dim=None, out_dim=None, activation=nn.GELU, drop=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        dim = dim or in_dim
        self.fc1 = nn.Linear(in_dim, dim)
        self.act = activation()
        self.fc2 = nn.Linear(dim, out_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, nheads, qkv_bias=False, qk_scale=None, attn_drop=0., mlp_ratio=2., drop=0.1, activation=nn.GELU, normlayer=nn.LayerNorm):
        super().__init__()
        self.norm1 = normlayer(dim)
        self.attention = Attention(dim, nheads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = normlayer(dim)
        self.mlp = MLP(in_dim=dim, dim=int(dim*mlp_ratio), activation=activation, drop=drop)

    def forward(self, x, mask=None):
        x = self.norm1(x)
        x, attn = self.attention(x, mask)
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class SENvT(nn.Module):
    def __init__(
            self, dim=1024, nheads=16, nlayers=24,
            window_size=300, in_chans=3, patch_size=1, num_classes=0,
            is_eva=False, is_rotation_task=False, augment_chunk_size=50,
        **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.patch_embed = PatchEmbed(dim, window_size, patch_size, in_chans, is_eva=is_eva)
        self.pos_embed = nn.Parameter(torch.rand(1, self.patch_embed.num_patches+1, dim), requires_grad=True)
        # self.mask_embed = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.nheads = nheads
        self.layers = nn.ModuleList([EncoderLayer(dim, nheads) for i in range(nlayers)])
        self.norm = nn.LayerNorm(dim, eps=1e-5, bias=True)
        self.pos_drop = nn.Dropout(p=0.)
        
        self.num_classes = num_classes
        if num_classes > 0 or is_eva:
            self.head = Classifier(dim, num_classes, npatches=self.patch_embed.num_patches)
        else:
            self.head = TimeSeriesHead(
                dim=dim, nheads=nheads, nlayers=2,
                window_size=window_size, in_chans=in_chans, patch_size=patch_size,
                is_rotation_task=is_rotation_task, delta=augment_chunk_size
            )
    
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def prepare_tokens(self, x, mask=None):
        N,C,L = x.shape
        x, mask = self.patch_embed(x, mask=mask)
        
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x, mask
    
    def forward(self, x, mask=None):
        x, mask = self.prepare_tokens(x, mask=mask)
        # encoder
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        
        # head
        if self.num_classes > 0:
            return self.head(x[:, 0])
        return self.head(x[:, 1:])


class TimeSeriesHead(nn.Module):
    def __init__(
        self, dim=1024, nheads=16, nlayers=24, window_size=300, in_chans=3, patch_size=1,
        is_rotation_task=False, delta=50,
    ):
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=True)
        
        self.in_chans = in_chans
        npatches = window_size // patch_size
        
        self.pos_embed = nn.Parameter(torch.zeros(1, npatches, dim), requires_grad=True)
        self.layers = nn.ModuleList([EncoderLayer(dim=dim, nheads=nheads) for i in range(2)])
        self.pred = nn.Linear(dim, patch_size*in_chans, bias=True)
        
        """ rotation head:
        rotation タスクは複数チャネルの場合，チャネルのランダム変換が適用される．
        チャネルのクラス分類タスクとして考えるので，損失計算はMSEではなく，CorssEntoropyの方が好ましいと考えた．
        そのためのrotation headである．
        """
        self.is_rotation_task = is_rotation_task
        if self.is_rotation_task:
            dpatches = delta // patch_size
            self.rotation_linear = nn.Linear(dim, dim)
            self.rotation_proj = nn.Conv1d(dim, in_chans**2, kernel_size=dpatches, stride=dpatches)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.embed(x)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        rh = self.rotation_head(x) if self.is_rotation_task else None
        
        x = self.pred(x)
        x = x.reshape(B, -1, self.in_chans)
        x = x.permute(0, 2, 1) # BWC → BCW
        return x, rh
    
    def rotation_head(self, x):
        B,L,D = x.shape
        x = self.rotation_linear(x)
        x = F.relu(x)
        x = x.transpose(1,2)
        x = self.rotation_proj(x)
        x = x.transpose(1,2)
        x = x.reshape(B, -1, self.in_chans, self.in_chans)
        return x


class Classifier(nn.Module):
    def __init__(self, dim=1024, num_classes=2, npatches=300):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

def XS(**args):
    return SENvT(dim=192, nheads=3, nlayers=4, **args)
def S(**args):
    return SENvT(dim=384, nheads=6, nlayers=12, **args)
def B(**args):
    return SENvT(dim=768, nheads=12, nlayers=12, **args)
def L(**args):
    return SENvT(dim=1024, nheads=16, nlayers=24, **args)
def XL(**args):
    return SENvT(dim=1280, nheads=20, nlayers=32, **args)
    