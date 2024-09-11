
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import torch
import math

import pdb

# Split Last Dimension (channel dimension)
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

# Restore the last dimension (channel dimension)
def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

# Add 3D unilateral zero data for conv2×2×2 (1,0,1,0,1,0)
class CustomPad(torch.nn.Module):
  def __init__(self, padding):
      super(CustomPad, self).__init__()
      self.padding = padding
  def forward(self, x):
      return F.pad(x,self.padding, mode='constant',value=0)

# MHSA and local feature extraction
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, device):
        super().__init__()

        self.device = device

        self.proj_q = nn.Sequential(
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            nn.Conv3d(dim, dim, 2, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),  
            #nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.pad = CustomPad(padding=(1,0,1,0,1,0))
        #self.conv
        #self.proj_q = nn.Linear(dim, dim)
        #self.proj_k = nn.Linear(dim, dim)
        #self.proj_v = nn.Linear(dim, dim)
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization


    def forward(self, x ):    # [B, 192, 32]
        """
        x, q(query), k(key), v(value) : (B(batch_size), P(T*H*W), D(dim))
        * split C(dim) into (n(n_heads), w(width of head)) ; C=32
        """
        [B, P, C]=x.shape   # B(batch_size), P(T*H*W), D(dim)
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 12, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(self.pad(x)), self.proj_v(x)


        q_se,k_se,v_se = q,k,v  #  q,k,v: [B,C,T,H,W]]
        # local feature extraction
        q_se, k_se, v_se = (split_last(x.permute(0, 2, 3, 4, 1), (self.n_heads, -1)).permute(0, 4, 5, 1, 2, 3) for x in
                            [q_se, k_se, v_se])  # q_se:[B,C,T,H,W]->[B,T,H,W,C]->[B,T,H,W,n,w]->[B,n,w,T,H,W]
        B, h, w, T, H, W = q_se.shape

        # concat
        conv_atten = torch.zeros((B, h, 3 * w, T, H, W)).to(self.device)   # [B,h,3*w,T,H,W]
        conv_atten[:, :, 0::3, :, :, :] = q_se
        conv_atten[:, :, 1::3, :, :, :] = k_se
        conv_atten[:, :, 2::3, :, :, :] = v_se
        # Avg pooling
        conv_atten = conv_atten.view(B, h, 3, w, T, H, W)   # [B,h,3,w,T,H,W]
        conv_atten = torch.squeeze(torch.mean(conv_atten, 2),dim=2)  # [B,h,w,T,H,W]
        conv_atten = conv_atten.flatten(3).permute(0,3,1,2)   # [B,T*H*W,h,w]
        
        # MHSA
        q = q.flatten(2).transpose(1, 2)  # [B, T*H*W, C]
        k = k.flatten(2).transpose(1, 2)  # [B, T*H*W, C]
        v = v.flatten(2).transpose(1, 2)  # [B, T*H*W, C]

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, h, P, w) @ (B, h, w, P) -> (B, h, P, P) -softmax-> (B, h, P, P)
        scores = q @ k.transpose(-2, -1)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, h, P, P) @ (B, h, P, w) -> (B, h, P, w) -trans-> (B, P, h, w)
        h = (scores @ v).transpose(1, 2).contiguous()
        # local feature extraction + MHSA
        h = h + conv_atten
        h = merge_last(h, 2)
        self.scores = scores
        return h

# Muti-Scale Spatialtemporal Feed-forward
class MSSTFF(nn.Module):

    def __init__(self):
        super().__init__()
        self.ST_1 = nn.Sequential(
            nn.Conv3d(1, 1, 1, stride=1, padding=0, bias=False),
        )
        self.pad = CustomPad(padding=(1, 0, 1, 0, 1, 0))
        self.ST_2 = nn.Sequential(
            nn.Conv3d(1, 1, 2, stride=1, padding=0, bias=False),
        )
        self.ST_3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, P, C = x.size()  # [B, 192, 32]
        x = x.permute(0, 2, 1).view(B, C, P//16, 4, 4)
        n, c, t, h, w = x.size()  # [B,32,12,4,4]
        x_shift = x.permute(0,2,1,3,4).view(n*t,c,h,w)  # [B*12,32,4,4]
        x = x.mean(1, keepdim=True)  # [B,1,12,4,4]
        x_p1 = self.ST_1(x)  # [B,1,12,4,4]
        x_p2 = self.ST_2(self.pad(x))  # [B,1,12,4,4]
        x_p3 = self.ST_3(x)  # [B,1,12,4,4]
        x = x_p1+x_p2+x_p3   # [B,1,12,4,4]
        x = x.transpose(2, 1).contiguous().view(n*t, 1, h, w)  # [B*12,1,4,4]
        x_sig = self.sigmoid(x)
        x = x_shift * x_sig  # [B*12,32,4,4]
        return x.view(n,t,c,4,4).permute(0,2,1,3,4).view(n,c,-1).permute(0,2,1)
        # [B*12,32,4,4] -> [B,12,32,4,4] -> [B,32,12,4,4] -> [B,32,192] -> [B,192,32]

# Muti Scale Global Feauture Aggregation
class MSGFA(nn.Module):
    def __init__(self,dim):
        super(MSGFA,self).__init__()
        self.pad = CustomPad(padding=(1,0,1,0,1,0))
        self.resol_1 = nn.Sequential(
            nn.Conv3d(dim , dim, 3, stride=1, padding=1, groups=1, bias=False),
        )
        self.resol_2 = nn.Sequential(
            nn.Conv3d(dim, dim, 2, stride=1, padding=0, groups=1, bias=False),
        )
        self.resol_3 = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),

        )

    def forward(self,x):
        B, P, C = x.size()
        x = x .view(B, P // 16, 4, 4, C).permute(0,4,1,2,3)  # [B,32,16,4,4]
        # print(x.shape)
        conv_atten1 = conv_atten = self.resol_1(x) + x   # [B,32,16,4,4]
        conv_atten2 = conv_atten = self.resol_2(self.pad(conv_atten)) + conv_atten   # [B,32,16,4,4]
        conv_atten3 = conv_atten = self.resol_3(conv_atten) + conv_atten   # [B,32,16,4,4]
        conv_atten = torch.cat([conv_atten1, conv_atten2, conv_atten3], dim=1)   # [B,32*3,16,4,4]
        conv_atten = conv_atten.view(B,C,3,P//16,4,4)   # [B,32,3,16,4,4]
        conv_atten = torch.squeeze(torch.mean(conv_atten, 2),dim=2)   # [B,32,16,4,4]
        conv_atten = conv_atten.flatten(2).permute(0,2,1)  # [B,192,32]
        return conv_atten

class Block_LMTformer(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads,dropout, device):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout,device=device)  # MSMHSA(Muti-Scale MHSA)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.ST_atten = MSSTFF()  # MSSTFF(Muti-Scale Spatialtemporal Feed-forward)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.msgfa = MSGFA(dim=dim)   # MSGFA(Muti-Scale Global Feature Aggregation)

    def forward(self, x):
        Atten = self.attn(self.norm1(x))
        h = self.drop(self.proj(Atten))
        s = self.msgfa(x)
        #print(s.shape)
        x = s + h
        h = self.drop(self.ST_atten(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer×N"""
    def __init__(self, num_layers, dim, num_heads,dropout,device):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_LMTformer(dim, num_heads, dropout,device=device) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x