import torch
import torch.nn as nn
from  . import common

from einops import rearrange
from hydra.utils import instantiate
  
class I2IwFiLM_GVP_P(nn.Module):
    def __init__(self,inp_channels=3, gt_channels=3, n_feats = 64, n_encoder_res = 6, scale=4):
        super(I2IwFiLM_GVP_P, self).__init__()
        self.scale=scale

        input_channels = (gt_channels + inp_channels) * (scale ** 2) # (1+1) * (4 ** 2) = 32
        E1=[nn.Conv2d(input_channels, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)] # 32 -> 64 = C'
        
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ] # 64 -> 64 = C'

        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ] # 64 -> 64 * 4 = 256 = 4.C'
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        
        feat = self.pixel_unshuffle(x)

        x = torch.cat([feat, gt0], dim=1)

        fea = self.E(x).squeeze(-1).squeeze(-1)

        S1_IPR = []
        fea1 = self.mlp(fea)
        S1_IPR.append(fea1)
        return fea1, S1_IPR
  
class I2IwFiLM_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_film=False):
        super(I2IwFiLM_TransformerBlock, self).__init__()
        self.norm1 = common.LayerNorm(dim, LayerNorm_type)
        self.norm2 = common.LayerNorm(dim, LayerNorm_type)
        if use_film:
            self.attn = common.Attention_DA_FILM(dim, num_heads, bias)
            self.ffn = common.FeedForward_DFFN_FILM(dim, ffn_expansion_factor, bias)
        else:
            self.attn = common.Attention_DA(dim, num_heads, bias)
            self.ffn = common.FeedForward_DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        x = x + self.attn(self.norm1(x),k_v)
        x = x + self.ffn(self.norm2(x),k_v)

        return [x,k_v]

class ModularI2IwFiLMFormer(nn.Module):
    def __init__(self, 
        inp_channels=3,
        out_channels=3, 
        patch_size=4,
        scale=4,
        patchifier_embed_dim = 48,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        use_film=False,
        ):
        super(ModularI2IwFiLMFormer, self).__init__()
        self.scale=scale
        self.patch_size = patch_size
        
        self.num_blocks = num_blocks
        self.heads = heads

        print(f"inp_channels: {inp_channels}({type(inp_channels)}), patch_size: {patch_size}({type(patch_size)}), patchifier_embed_dim: {patchifier_embed_dim}({type(patchifier_embed_dim)})")

        self.patchifier = common.Patchifier(inp_channels, patch_size=self.patch_size, embed_dim=patchifier_embed_dim)  

        self.patch_embed = common.OverlapPatchEmbed(patchifier_embed_dim, dim)

        self.encoder_levels = nn.ModuleList([nn.Sequential(
                                                *[I2IwFiLM_TransformerBlock(dim=dim*2**i, num_heads=heads[i], 
                                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                                            bias=bias, LayerNorm_type=LayerNorm_type,
                                                                            use_film=use_film
                                                                            ) 
                                                for j in range(num_blocks[i])]
                                            ) for i in range(len(num_blocks))])
        self.downs = nn.ModuleList([common.Downsample(int(dim*2**i)) for i in range(0,len(num_blocks)-1)])
        self.ups = nn.ModuleList(reversed([common.Upsample(int(dim*2**i)) for i in range(len(num_blocks)-1,0,-1)]))

        self.reduce_chans = nn.ModuleList(reversed([nn.Conv2d(int(dim*2**i), int(dim*2**(i-1)), kernel_size=1, bias=bias) for i in range(len(num_blocks)-1,0,-1)]))
        
        self.decoder_levels = nn.ModuleList(reversed([nn.Sequential(
                                                            *[I2IwFiLM_TransformerBlock(dim=int(dim*2**i), num_heads=heads[i-1],
                                                                                    ffn_expansion_factor=ffn_expansion_factor, 
                                                                                    bias=bias, LayerNorm_type=LayerNorm_type,
                                                                                    use_film=use_film
                                                                                    )
                                                            for j in range(num_blocks[i-1])]) 
                                                     for i in range((len(num_blocks)-1)-1,0,-1)] 
                                                     +
                                                     [nn.Sequential(
                                                            *[I2IwFiLM_TransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                                                                    ffn_expansion_factor=ffn_expansion_factor, 
                                                                                    bias=bias, LayerNorm_type=LayerNorm_type,
                                                                                    use_film=use_film
                                                                                    )
                                                            for j in range(num_blocks[0])])]
                                                ))

        self.refinement = nn.Sequential(*[I2IwFiLM_TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], 
                                                                   ffn_expansion_factor=ffn_expansion_factor, 
                                                                   bias=bias, LayerNorm_type=LayerNorm_type,
                                                                   use_film=use_film) for i in range(num_refinement_blocks)])

        self.unpatchifier = common.Unpatchifier_V3(in_c=int(dim*2**1), out_c = out_channels,
                                                patch_size=patch_size, bias=bias, tanh = True)
        

    def forward(self, inp_img, k_v):

        feat = inp_img # (B, C, 256, 256)
        patched_feat = self.patchifier(feat) # (B, 48, 64, 64)
        inp_enc_level1 = self.patch_embed(patched_feat) # (B, 48, 64, 64)

        cur_level_enc_inp = inp_enc_level1
        # Downwards pass
        out_enc_levels = []
        for i in range(len(self.num_blocks)):
            out_enc_level,_ = self.encoder_levels[i]([cur_level_enc_inp ,k_v])
            out_enc_levels.append(out_enc_level)
            if i < len(self.num_blocks)-1:
                cur_level_enc_inp = self.downs[i](out_enc_level)

        latent = out_enc_levels[-1]

        cur_level_dec_inp = self.ups[-1](latent)
        # Upwards pass
        out_dec_levels = []
        for i in range((len(self.num_blocks)-1),0,-1):
            cur_level_cat = torch.cat([cur_level_dec_inp, out_enc_levels[i-1]], 1)
            if i > 1:
                cur_level_reduced = self.reduce_chans[i-1](cur_level_cat)
                out_dec_level,_ = self.decoder_levels[i-1]([cur_level_reduced,k_v])
                out_dec_levels.append(out_dec_level)
                cur_level_dec_inp = self.ups[(i-1)-1](out_dec_level)
            else:
                out_dec_level,_ = self.decoder_levels[i-1]([cur_level_cat,k_v])
                out_dec_levels.append(out_dec_level)
                cur_level_dec_inp = out_dec_level

        out_dec_level1,_ = self.refinement([out_dec_levels[-1],k_v])
        
        # go back to original channels and resolution
        out = self.unpatchifier(out_dec_level1)
        
        ### IN V2 WE DO NOT ADD THE INPUT IMAGE TO THE OUTPUT
        out_dec_level1 = out


        return out_dec_level1
 
class I2IwFiLM_S1(nn.Module):
    def __init__(self, 
        n_encoder_res=6,         
        inp_channels=3, 
        out_channels=3, 
        scale=4,
        patch_size=4,
        patchifier_embed_dim = 24,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        use_film=False,
        ):
        super(I2IwFiLM_S1, self).__init__()

        # Generator
        self.G = ModularI2IwFiLMFormer(        
            inp_channels=inp_channels, 
            out_channels=out_channels,
            scale = scale, 
            patch_size= patch_size,
            patchifier_embed_dim=patchifier_embed_dim,
            dim = dim,
            num_blocks = num_blocks, 
            num_refinement_blocks = num_refinement_blocks,
            heads = heads,
            ffn_expansion_factor = ffn_expansion_factor,
            bias = bias,
            LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
            use_film=use_film
        )

        self.E = I2IwFiLM_GVP_P(inp_channels=inp_channels, gt_channels=out_channels,
                                  n_feats=64, n_encoder_res=n_encoder_res,scale=scale)


    def forward(self, x, gt):
        if self.training:
            IPRS1, S1_IPR = self.E(x,gt)
            sr = self.G(x, IPRS1)

            return sr, S1_IPR
        else:
            IPRS1, S1_IPR = self.E(x,gt)
            sr = self.G(x, IPRS1)

            return sr, S1_IPR