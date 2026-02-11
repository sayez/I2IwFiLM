import torch
import torch.nn as nn

from  . import common
from . import S1_models as S1
  
class I2IwFiLM_GVP_W(nn.Module):
    def __init__(self,inp_channels=3, n_feats = 64, n_encoder_res = 6, scale=4):
        super(I2IwFiLM_GVP_W, self).__init__()
        self.scale=scale

        input_channels = (inp_channels) * (scale ** 2) # (1+1) * (4 ** 2) = 32
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

    def forward(self, x):    
        feat = self.pixel_unshuffle(x)
        
        fea = self.E(feat).squeeze(-1).squeeze(-1)

        fea1 = self.mlp(fea)
       
        return fea1
    
class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea

class GVP_P_to_GVP_W_MLP(nn.Module):
    def __init__(self,  
                    mlp_in_feats=64, # num channels in CPEN_S1 vectors
                    mlp_out_feats=64, # num channels in CPEN_S2 vectors
                    mlp_hidden=[64,64,64], # size of hidden layers ()
                    mlp_bias=False
                   ):
        super(GVP_P_to_GVP_W_MLP, self).__init__()

        self.in_feats = mlp_in_feats * 4 # times 4 as in the CPEN_S2
        self.out_feats = mlp_out_feats * 4 # times 4 as in the CPEN_S1
        self.hidden = [ h*4 for h in mlp_hidden] # times 4 as in the CPEN_S1
        self.bias = mlp_bias

        self.in_layer = [nn.Linear(self.in_feats, self.hidden[0], bias=self.bias)]
        self.hidden_layers = [nn.Linear(self.hidden[i], self.hidden[i+1], bias=self.bias) for i in range(len(self.hidden)-1)]
        self.out_layer = [nn.Linear(self.hidden[-1], self.out_feats, bias=self.bias)]

        self.hidden = []
        for i in range(len(self.hidden)-1):
            self.hidden.append(self.hidden_layers[i])
            self.hidden.append(nn.LeakyReLU(0.1, True))
        
        layers = self.in_layer + [nn.LeakyReLU(0.1, True)] + self.hidden + self.out_layer

        self.mlp = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.mlp(x)

class I2IwFiLM_S2(nn.Module):
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

        # mlp_in_feats=64, # num channels in CPEN_S2 vectors
        # mlp_out_feats=64, # num channels in CPEN_S1 vectors
        mlp_hidden=[64,64,64], # size of hidden layers ()

        zero_condition = False,

        ):
        super(I2IwFiLM_S2, self).__init__()

        # Generator
        self.STM = S1.ModularI2IwFiLMFormer(        
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
        )

        self.zero_condition = zero_condition
        self.n_feats = 64
        self.cpen_out_dim = self.n_feats * 4

        if not self.zero_condition:
            self.condition = I2IwFiLM_GVP_W(inp_channels=inp_channels, n_feats=64, n_encoder_res=n_encoder_res,scale = scale)

            self.cpens2_to_cpens1 = GVP_P_to_GVP_W_MLP(
                                        # mlp_in_feats=mlp_in_feats, # num channels in CPEN_S2 vectors
                                        # mlp_out_feats=mlp_out_feats, # num channels in CPEN_S1 vectors
                                        mlp_hidden=mlp_hidden, # size of hidden layers ()
                                        )

    def forward(self, img):
        if self.zero_condition:
            Ew = torch.zeros(img.shape[0], self.cpen_out_dim).to(img.device)
            Ep_hat = torch.zeros(img.shape[0], self.cpen_out_dim).to(img.device) 
            sr = self.STM(img, Ep_hat)

        else:
            # Phase 1
            # get the input vector of the MLP, from GVP_W
            Ew = self.condition(img) # D
            # get the output vector of the MLP, reconstruction of the GVP_P vector
            Ep_hat = self.cpens2_to_cpens1(Ew)

            # Phase 2
            # From input image and IPRS2, get output image
            # (img, IPRS2) -|I2IwFiLMFormer|-> output
            sr = self.STM(img, Ep_hat)

        return sr, Ep_hat, Ew