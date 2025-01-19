"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
import math
import torch.nn.init as init


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)

        self.fuser8 = FeatureFusionBlock(512, 128, 256, 256)
        self.fuser4 = FeatureFusionBlock(256, 64, 128, 128)
        
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image,
                image_feat_f16, image_feat_f8, image_feat_f4,
                h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g4 = self.layer1(g) # 1/4
        g8 = self.layer2(g4) # 1/8
        g16 = self.layer3(g8) # 1/16

        g16 = g16.view(batch_size, num_objects, *g16.shape[1:])
        g16 = self.fuser(image_feat_f16, g16)

        g8 = g8.view(batch_size, num_objects, *g8.shape[1:])
        g8 = self.fuser8(image_feat_f8, g8)
        g4 = g4.view(batch_size, num_objects, *g4.shape[1:])
        g4 = self.fuser4(image_feat_f4, g4)
        
        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g16, h)

        return g16, g8, g4, h
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        print(g.shape)
        g = self.out_conv(g)
        return g

class scale_UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.mem_conv = nn.Conv2d(g_up_dim//2, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g, mem_f):
        skip_f = self.skip_conv(skip_f)
        b, no = mem_f.shape[:2]
        mem_f = self.mem_conv(mem_f.flatten(start_dim=0, end_dim=1))
        mem_f = mem_f.view(b, no, *mem_f.shape[1:])
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = g + mem_f
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g
    

class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()
        self.attn1 = FATBlock(512, 512, 9, 16, 2, 4)
        self.attn2 = FATBlock(512, 512, 9, 16, 2, 4)
        self.key_proj_1 = nn.Conv2d(in_dim, 512, kernel_size=3, padding=1) # 1024->512
        self.key_proj_2 = nn.Conv2d(512, keydim, kernel_size=3, padding=1) # 512->64
        nn.init.orthogonal_(self.key_proj_1.weight.data)
        nn.init.zeros_(self.key_proj_1.bias.data)
        nn.init.orthogonal_(self.key_proj_2.weight.data)
        nn.init.zeros_(self.key_proj_2.bias.data)

        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        x_mid = self.key_proj_1(x)    # 1024-> 512
        x_attn1 = self.attn1(x_mid)   # global & local
        x_attn2 = self.attn2(x_attn1) # global & local

        return self.key_proj_2(x_attn2), shrinkage, selection

class KeyProjection_Scale(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
    
class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None

        self.scale_up_16_8 = scale_UpsampleBlock(512, 512, 256)
        self.scale_up_8_4 = scale_UpsampleBlock(256, 256, 256)
        
        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, 
                memory_readout, memory_readout_8, memory_readout_4,
                h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.scale_up_16_8(f8, g16, memory_readout_8)
        g4 = self.scale_up_8_4(f4, g8, memory_readout_4)     

        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class Get_Sr_Ratio(nn.Module):
    def __init__(self, input_channels, N):
        super(Get_Sr_Ratio, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(2)
        self.linear1 = nn.Linear(2 * 2 * input_channels, 128)
        self.linear2 = nn.Linear(128, N)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the adaptive pool
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.softmax(x, 1)
    
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Dynamic_conv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sbn = nn.SyncBatchNorm(out_planes)

        self.weight_2 = nn.Parameter(torch.Tensor(out_planes, in_planes, 2, 2), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        self.weight_4 = nn.Parameter(torch.Tensor(out_planes, in_planes, 4, 4), requires_grad=True)
        self.bias_4 = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_4, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight_2[0, 0].numel())
        nn.init.uniform_(self.bias_2, -bound, bound)
        bound = 1 / math.sqrt(self.weight_4[0, 0].numel())
        nn.init.uniform_(self.bias_4, -bound, bound)


    def forward(self, x, ratio_sr):
        ratio_index  =  torch.argmax(ratio_sr, dim=1)[0]
        softmax_attention = ratio_sr[0][ratio_index]

        if ratio_index == 0:
            return x, ratio_index
        elif ratio_index == 1:
            aggregate_weight = softmax_attention * self.weight_2
            aggregate_bias = softmax_attention * self.bias_2
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=2)
            return self.sbn(output), ratio_index
        
        elif ratio_index == 2:
            aggregate_weight = softmax_attention * self.weight_4
            aggregate_bias = softmax_attention * self.bias_4
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=4)
            return self.sbn(output), ratio_index
        
# 全局注意力
class FASA(nn.Module):

    def __init__(self, dim: int, kernel_size: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_head = num_heads
        self.dim_head = dim // num_heads
        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(self.dim_head *4 , self.dim_head * 2 * 4, 1, 1, 0)
        self.pool = Dynamic_conv2d(128, 128)
        self.attention = Get_Sr_Ratio(128, 3)
        
        self.window_size = window_size
        self.scalor = self.dim_head ** -0.5
        # local branch
        self.local_mixer1 = nn.Sequential(
                get_dwconv(dim, 3, True),
                nn.SyncBatchNorm(dim),
                nn.ReLU(inplace = True)
            )
        self.local_mixer2 = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.SyncBatchNorm(dim)
            )
        self.global_mixer = nn.Conv2d(dim, dim, 1, 1, 0)

    def refined_downsample(self, dim, window_size, kernel_size):
        if window_size==1:
            return nn.Identity()
        for i in range(4):
            if 2**i == window_size:
                break
        block = nn.Sequential()
        for num in range(i):
            block.add_module('conv{}'.format(num), nn.Conv2d(dim, dim, kernel_size, 2, kernel_size//2, groups=dim))
            block.add_module('bn{}'.format(num), nn.SyncBatchNorm(dim))
            if num != i-1:
                block.add_module('linear{}'.format(num), nn.Conv2d(dim, dim, 1, 1, 0))
        return block

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        q_local = self.q(x)
        tensor_list_head = []
        tensor_list_batch = []
        # local
        local_feat = self.local_mixer1(q_local) 
        local_feat = self.local_mixer2(local_feat)
        # global
        for i in range (b):    
            x_sample = x[i].reshape(1, c, h, w)
            
            q4ratio = q_local[i].reshape(-1, self.dim_head * 4, h, w).contiguous()
            ratio = self.attention(q4ratio)

            for j in range(self.num_head // 4):
                group_x = x_sample[-1, self.dim_head * j * 4 : self.dim_head * (j + 1) * 4].reshape(1, -1, h, w)
                pool_x, index = self.pool(group_x, ratio[j].reshape(-1, 3))
                _, _, h_down, w_down = pool_x.shape
                k, v = self.kv(pool_x).reshape(1, 2, -1, self.dim_head*4, h_down * w_down).permute(1, 0, 2, 4, 3).contiguous()
                q4attn = q_local[i, self.dim_head * j * 4: self.dim_head * (j + 1) * 4 ].reshape(1, -1, self.dim_head * 4, h * w).transpose(-1, -2).contiguous()

                attn = torch.softmax(q4attn @ k.transpose(-1, -2), -1)
                global_feat_j = attn @ v
                global_feat_j = global_feat_j.transpose(-1, -2).reshape(1, 32 * 4, h, w)
                tensor_list_head.append(global_feat_j)
            global_feat_head = torch.cat(tensor_list_head, dim=1)
            global_feat_head = self.global_mixer(global_feat_head)
            tensor_list_head.clear()
            tensor_list_batch.append(global_feat_head)
        global_feat = torch.cat(tensor_list_batch, dim=0)

        return local_feat * global_feat
        
    
class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels,  out_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x) #(b c h w)
        return x 
    

class FATBlock(nn.Module):

    def __init__(self, dim: int, out_dim: int, kernel_size: int, num_heads: int, window_size: int, 
                 mlp_ratio: float):
        super().__init__()
        self.dim = dim

        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = FASA(dim, kernel_size, num_heads, window_size)
        self.drop_path = nn.Identity()
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.ffn = ConvFFN(dim, mlp_hidden_dim, out_dim) # FeedForward

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x