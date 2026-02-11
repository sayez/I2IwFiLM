import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers

from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

    
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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    


class FeedForward_DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DFFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hidden_features = hidden_features

        self.Z_linear = nn.Sequential(
                nn.Linear(256, dim, bias=False),
            )

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        # self.dwconv2 = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x, k_v):
        b,c,h,w = x.shape
        # print()
        # print(f'FeedForward_DFFN hidden_features: {self.hidden_features}')
        # print(f'FeedForward_DFFN input shape: {x.shape}, k_v shape: {k_v.shape}')
        x_skip = x # Shape : H,W,C

        # a) path for F                
        x_tmp = self.project_in(x) # Shape : H,W,C*2
        # print(f'FeedForward_DFFN x_tmp shape after project_in: {x_tmp.shape}')
        x_tmp = self.dwconv1(x_tmp) # Shape : H,W,C*2
        # print(f'FeedForward_DFFN x_tmp shape after dwconv1: {x_tmp.shape}')

        # Apply SimpleGate
        x_1, x_2 = x_tmp.chunk(2, dim=1)
        x_tmp = x_1 * x_2 # Shape : H,W,C
        # print(f'FeedForward_DFFN x_tmp shape after SimpleGate: {x_tmp.shape}')

        x_tmp = self.dwconv2(x_tmp) # Shape : H,W,C
        # print(f'FeedForward_DFFN x_tmp shape after dwconv2: {x_tmp.shape}')
        x_tmp = self.project_out(x_tmp) # Shape : H,W,C

        # b) path for Z
        k_v=self.Z_linear(k_v).view(-1,c,1,1) # Shape : H,W,C
        # print(f'FeedForward_DFFN k_v shape after Z_linear: {k_v.shape}')

        # c) merge
        x_tmp = x_tmp + k_v
        # print(f'FeedForward_DFFN x_tmp shape after merge: {x_tmp.shape}')
        x = x_skip + x_tmp # Shape : H,W,C
        # print(f'FeedForward_DFFN output shape: {x.shape}')
        return x

class FeedForward_DFFN_FILM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DFFN_FILM, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hidden_features = hidden_features

        self.Z_linear = nn.Sequential(
                nn.Linear(256, dim*2, bias=False),
            )

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        # self.dwconv2 = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x, k_v):
        b,c,h,w = x.shape
        # print()
        # print(f'FeedForward_DFFN hidden_features: {self.hidden_features}')
        # print(f'FeedForward_DFFN input shape: {x.shape}, k_v shape: {k_v.shape}')
        x_skip = x # Shape : H,W,C

        # a) path for F                
        x_tmp = self.project_in(x) # Shape : H,W,C*2
        # print(f'FeedForward_DFFN x_tmp shape after project_in: {x_tmp.shape}')
        x_tmp = self.dwconv1(x_tmp) # Shape : H,W,C*2
        # print(f'FeedForward_DFFN x_tmp shape after dwconv1: {x_tmp.shape}')

        # Apply SimpleGate
        x_1, x_2 = x_tmp.chunk(2, dim=1)
        x_tmp = x_1 * x_2 # Shape : H,W,C
        # print(f'FeedForward_DFFN x_tmp shape after SimpleGate: {x_tmp.shape}')

        x_tmp = self.dwconv2(x_tmp) # Shape : H,W,C
        # print(f'FeedForward_DFFN x_tmp shape after dwconv2: {x_tmp.shape}')
        x_tmp = self.project_out(x_tmp) # Shape : H,W,C

        # b) path for Z
        k_v=self.Z_linear(k_v).view(-1,c,1,1) # Shape : H,W,C
        # print(f'FeedForward_DFFN k_v shape after Z_linear: {k_v.shape}')

        # split k_v into gamma and beta
        gamma, beta = k_v.chunk(2, dim=1)

        # c) Apply FiLM Modulation: element-wise multiplication and addition
        x_tmp = x_tmp * gamma + beta

        # print(f'FeedForward_DFFN x_tmp shape after merge: {x_tmp.shape}')
        x = x_skip + x_tmp # Shape : H,W,C
        # print(f'FeedForward_DFFN output shape: {x.shape}')
        return x

class FeedForward_DGFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DGFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2  # F' 
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class Attention_DA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_DA, self).__init__()

        self.Z_linear = nn.Sequential(
            nn.Linear(256, dim, bias=False),
        )
        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, k_v):
        # print()
        # print(f'Attention_DA input shape: {x.shape}, k_v shape: {k_v.shape}')
        b,c,h,w = x.shape
        x_skip = x # Shape : H,W,C
        # a) path for F
        x_tmp = self.project_in(x)
        # print(f'Attention_DA x_tmp shape: {x_tmp.shape}')
        x_tmp = self.dwconv1(x_tmp) # Shape : H,W,C*2
        # print(f'Attention_DA x_tmp shape after dwconv1: {x_tmp.shape}')
        # Apply SimpleGate
        x_1, x_2 = x_tmp.chunk(2, dim=1)
        F_sg = x_1 * x_2 # Shape : H,W,C
        # print(f'Attention_DA F_sg shape: {F_sg.shape}')
        F_phi = self.avg_pool(F_sg) # Shape : 1,1,C
        # print(f'Attention_DA F_phi shape: {F_phi.shape}')
        F_ca = F_sg * F_phi # Shape : H,W,C
        # print(f'Attention_DA F_ca shape: {F_ca.shape}')
        x_tmp = self.dwconv2(F_ca) # Shape : H,W,C
        # print(f'Attention_DA x_tmp shape after dwconv2: {x_tmp.shape}')
        # b) path for Z
        k_v=self.Z_linear(k_v).view(-1,c,1,1) # Shape : H,W,C
        # print(f'Attention_DA k_v shape after Z_linear: {k_v.shape}')
        # c) merge
        x_tmp = x_tmp + k_v
        # print(f'Attention_DA x_tmp shape after merge: {x_tmp.shape}')
        x = x_skip + x_tmp # Shape : H,W,C
        # print(f'Attention_DA output shape: {x.shape}')

        return x
class Attention_DA_FILM(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_DA_FILM, self).__init__()

        self.Z_linear = nn.Sequential(
            nn.Linear(256, 2*dim, bias=False),
        )
        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, k_v):
        # print()
        # print(f'Attention_DA input shape: {x.shape}, k_v shape: {k_v.shape}')
        b,c,h,w = x.shape
        x_skip = x # Shape : H,W,C
        # a) path for F
        x_tmp = self.project_in(x)
        # print(f'Attention_DA x_tmp shape: {x_tmp.shape}')
        x_tmp = self.dwconv1(x_tmp) # Shape : H,W,C*2
        # print(f'Attention_DA x_tmp shape after dwconv1: {x_tmp.shape}')
        # Apply SimpleGate
        x_1, x_2 = x_tmp.chunk(2, dim=1)
        F_sg = x_1 * x_2 # Shape : H,W,C
        # print(f'Attention_DA F_sg shape: {F_sg.shape}')
        F_phi = self.avg_pool(F_sg) # Shape : 1,1,C
        # print(f'Attention_DA F_phi shape: {F_phi.shape}')
        F_ca = F_sg * F_phi # Shape : H,W,C
        # print(f'Attention_DA F_ca shape: {F_ca.shape}')
        x_tmp = self.dwconv2(F_ca) # Shape : H,W,C
        # print(f'Attention_DA x_tmp shape after dwconv2: {x_tmp.shape}')
        
        # b) path for Z
        k_v=self.Z_linear(k_v).view(-1,c,1,1) # Shape : H,W,C
        # print(f'Attention_DA k_v shape after Z_linear: {k_v.shape}')
        # split k_v into gamma and beta
        gamma, beta = k_v.chunk(2, dim=1)
       
        # c) Apply FiLM Modulation: element-wise multiplication and addition
        x_tmp = x_tmp * gamma + beta

        # print(f'Attention_DA x_tmp shape after merge: {x_tmp.shape}')
        x = x_skip + x_tmp # Shape : H,W,C
        # print(f'Attention_DA output shape: {x.shape}')

        return x


class Attention_DMTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_DMTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x,k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2  

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
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
 

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_DMTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_DGFN(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        x = x + self.attn(self.norm1(x),k_v)
        x = x + self.ffn(self.norm2(x),k_v)

        return [x,k_v]
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
class OverlapPatchRestore(nn.Module):
    def __init__(self, in_c=3, out_c=3, patch_size=16, bias=False):
        super(OverlapPatchRestore, self).__init__()

        kernel_size = (patch_size) + 1
        padding = (patch_size // 2)

        self.proj = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Patchifier(nn.Module):
    def __init__(self, in_c=3, patch_size=16, embed_dim=48, bias=False):
        super(Patchifier, self).__init__()

        kernel_size = (patch_size) + 1
        padding = (patch_size // 2)

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


def Log2(x):
    if x == 0:
        return False
 
    return (math.log10(x) /
            math.log10(2))
 
# Function to check
# if x is power of 2
  
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) ==
            math.floor(Log2(n)))

def powers_of_two_below_n(n):
    powers = []
    power_of_two = 1

    while power_of_two < n:
        powers.append(power_of_two)
        power_of_two *= 2

    return powers
    
class Unpatchifier(nn.Module):
    def __init__(self, in_c, out_c, patch_size, bias=False, tanh=True):
        super(Unpatchifier, self).__init__()

        # BxCx(H/p)x(W/p) -> BxCxHxW
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', )
        # same-size conv ( smoothing )
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        
        # BxCxHxW -> Bx1xHxW by 1x1 conv
        # in a for loop create
        self.in_channels = in_c
        self.pw2 = powers_of_two_below_n(in_c) # 
        # remove numbers under 32
        self.pw2 = reversed([x for x in self.pw2 if x >= 32])
        # print(f'pw2: {self.pw2}')
        pairs = [self.in_channels, *self.pw2]
        # print(f'pairs: {pairs}')
        self.channel_reduc_convs = nn.Sequential(*[nn.Conv2d(pairs[i], pairs[i+1], kernel_size=1, stride=1) for i in  range(len(pairs)-1)])

        # self.channel_reduc_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1) for i in range(3)])
        self.to_output_dim = nn.Conv2d(in_channels=pairs[-1], out_channels=out_c, kernel_size=1,
                               stride=1, padding=0)
        # self.fc = nn.Linear(in_features=in_c, out_features=out_c)

        self.use_tanh = tanh
        if self.use_tanh:
            # make sure the output is in the range of [-1, 1]
            self.tanh = nn.Tanh()

    def forward(self, x):
        # print(self)

        x = self.upsample(x)
        x = self.conv(x)
        x = self.channel_reduc_convs(x)

        x = self.to_output_dim(x)

        if self.use_tanh:
            x = self.tanh(x)

        return x
    

   
class Unpatchifier_V2(nn.Module):
    def __init__(self, in_c, out_c, patch_size, bias=False, tanh=True):
        super(Unpatchifier_V2, self).__init__()

        if patch_size > 1:
            # BxCx(H/p)x(W/p) -> BxCxHxW
            self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', )
        else:
            self.upsample = nn.Identity()
            
        # same-size conv ( smoothing )
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        
        # BxCxHxW -> Bx1xHxW by 1x1 conv
        # in a for loop create
        self.in_channels = in_c
        self.pw2 = powers_of_two_below_n(in_c) # 
        # remove numbers under 32
        self.pw2 = reversed([x for x in self.pw2 if x >= 32])
        # print(f'pw2: {self.pw2}')
        pairs = [self.in_channels, *self.pw2]
        # print(f'pairs: {pairs}')
        self.channel_reduc_convs = nn.Sequential(*[nn.Conv2d(pairs[i], pairs[i+1], kernel_size=1, stride=1) for i in  range(len(pairs)-1)])

        # self.channel_reduc_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1) for i in range(3)])
        self.to_output_dim = nn.Conv2d(in_channels=pairs[-1], out_channels=out_c, kernel_size=1,
                               stride=1, padding=0)
        # self.fc = nn.Linear(in_features=in_c, out_features=out_c)

        self.use_tanh = tanh
        if self.use_tanh:
            # make sure the output is in the range of [-1, 1]
            self.tanh = nn.Tanh()

    def forward(self, x):
        # print(self)

        x = self.upsample(x)
        x = self.conv(x)
        x = self.channel_reduc_convs(x)

        x = self.to_output_dim(x)

        if self.use_tanh:
            x = self.tanh(x)

        return x
    

class Unpatchifier_V3(nn.Module):
    def __init__(self, in_c, out_c, patch_size, bias=False, tanh=True):
        super(Unpatchifier_V3, self).__init__()

        # BxCx(H/p)x(W/p) -> BxCxHxW
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', )
        # same-size conv ( smoothing )
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        
        # BxCxHxW -> Bx1xHxW by 1x1 conv
        # in a for loop create
        self.in_channels = in_c
        self.pw2 = powers_of_two_below_n(in_c) # 
        # remove numbers under 32
        self.pw2 = reversed([x for x in self.pw2 if x >= 32])
        # print(f'pw2: {self.pw2}')
        pairs = [self.in_channels, *self.pw2]

        # print(f'pairs: {pairs}')
        conf_list = [nn.Conv2d(pairs[i], pairs[i+1], kernel_size=1, stride=1) for i in  range(len(pairs)-1)]
        # create activation for each layer
        for i in range(len(conf_list)):
            conf_list.insert(i*2+1, nn.LeakyReLU(0.1, inplace=True))

        print(f'conf_list: {conf_list}')

        self.channel_reduc_convs = nn.Sequential(*conf_list)

        # self.channel_reduc_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1) for i in range(3)])
        self.to_output_dim = nn.Conv2d(in_channels=pairs[-1], out_channels=out_c, kernel_size=1,
                               stride=1, padding=0)
        # self.fc = nn.Linear(in_features=in_c, out_features=out_c)

        self.use_tanh = tanh
        if self.use_tanh:
            # make sure the output is in the range of [-1, 1]
            self.tanh = nn.Tanh()

    def forward(self, x):
        # print(self)

        x = self.upsample(x)
        x = self.conv(x)
        x = self.channel_reduc_convs(x)

        x = self.to_output_dim(x)

        if self.use_tanh:
            x = self.tanh(x)

        return x
    