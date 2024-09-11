'''
Facial Depression Recognition with Lightweight Muti-Scale Transformer from Videos
'''
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange
from LMTformer_model.transformer_layer import Transformer
import pdb


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

# Multilayer perceptron
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# stem_3DCNN + LMTformer + Finnal BDI-II Prediction
class LMTformer(nn.Module):

    def __init__(
        self,
        device,
        name: Optional[str] = None, 
        pretrained: bool = False, 
        #patches: int = 16,
        patches=None,
        dim: int = 768,
        #ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        #theta: float = 0.2,
        #image_size: Optional[int] = None,
        image_size=None
    ):
        super().__init__()
        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes(24,128,128)
        ft, fh, fw = as_tuple(patches)  # patch sizes(2,4,4)
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    kernel_size=(2,4,4),stride=(2,4,4)
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                       dropout=dropout_rate,device=device)

        # Transformer
        self.transformer2 = Transformer(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                      dropout=dropout_rate,device=device)
        # Transformer
        self.transformer3 = Transformer(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                      dropout=dropout_rate, device=device)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = Mlp(in_features=dim,hidden_features=32,out_features=1,drop=0.2)
        #self.fc = fc(in_features=dim)
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim //2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.normLast = nn.LayerNorm(dim, eps=1e-6)
        self.bn = nn.BatchNorm3d(dim)

 
        #self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        #self.head = nn.Linear(dim,1)
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x):
        #print(x.shape)
        b, c, t, fh, fw = x.shape  # [B, 3, 24, 128, 128]
        # Coarse-grained Feature Extraction
        x = self.Stem0(x)  # [B, 8, 24, 64, 64]
        x = self.Stem1(x)  # [B, 16, 24, 32, 32]
        x = self.Stem2(x)  # [B, 32, 24, 16, 16]
        # Patch Embedding
        x = self.patch_embedding(x)  # [B, 32, 12, 4, 4]

        x = x.flatten(2).transpose(1, 2)  # [B, 192, 32]
        # Muti-Scale Transformer
        Trans_features =  self.transformer1(x)  # [B, 192, 32]
        Trans_features2 =  self.transformer2(Trans_features)  # [B, 192, 32]
        Trans_features3 =  self.transformer3(Trans_features2)  # [B, 192, 32]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//2, 4, 4)  # [B, 32, 12, 4, 4]
        # Final BDI-II Prediction
        x = self.pool(features_last)  # [B, 32, 1, 1, 1]
        x = x.view(b,-1)  # [B, 32]
        x = self.mlp(x)  # [B, 1]

        return x

if __name__ == '__main__':
    from thop import profile
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LMTformer(image_size=(24,128,128),patches=(2,4,4), dim=32, num_heads=4, num_layers=12, dropout_rate=0.1,device=device).to(device)
    #model.load_state_dict(torch.load('./model_dict_vst.pt',map_location='cuda:0'))
    input = torch.randn(1,3,24,128,128).to(device)
    out = model(input)
    flops, params = profile(model, inputs=(input,))
    print(flops/(1000**3),'G')
    print(params/(1000**2),'M')
    #print(model)

    #print(out.shape)
