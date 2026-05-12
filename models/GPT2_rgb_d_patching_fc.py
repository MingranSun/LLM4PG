import numpy as np
import torch
import torch.nn as nn
from torch import optim
import argparse

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import torch.nn as nn
import torch.nn.functional as F
import torch

class Frequency_fc(nn.Module):
    def __init__(self, configs, device):
        super(Frequency_fc, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 768)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Conv_patching_RGB(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_RGB, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 768
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 128*128
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('001:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('002:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
       # print('003:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        outputs = x.flatten(2).transpose(1, 2)
        #print('004:',outputs.shape)

        #print('005:',self.position_encoding.shape)
        
        outputs = outputs + self.position_encoding
        #print('006:',outputs.shape) # [batch_size, 64]
        return outputs


class Conv_patching_dep(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(Conv_patching_dep, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        

        num_patches = 64
        embed_dim = 768
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8) #RGB 256*256
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, max_num_patches, 768]

    def forward(self, x):
        #print('00:',x.shape)
        x = self.conv_patch(x)
        #print('01:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = self.bn_patch(x)
        #print('02:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true
        x = nn.functional.relu(x)
        #print('03:',x.shape)
        # print(f"x requires_grad: {x.requires_grad}") # true

        outputs = x.flatten(2).transpose(1, 2)
        #print('04:',outputs.shape)
        outputs = outputs + self.position_encoding

        #print('06:',outputs.shape) # [batch_size, 64]
        
        return outputs


class GPTRGBD(nn.Module): 
    def __init__(self, configs, device,normal_channel=False):
        super(GPTRGBD, self).__init__()
    
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
#        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(768, 65536)
        self.fc_compress = nn.Linear(129, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #(B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        

#        各层转移到device上
        for layer in (self.gpt2, self.in_layer, self.out_layer, self.fc_compress):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x):
#      
        batch_size, patches, patch_size = x.size()
        if self.is_gpt:

            x = self.gpt2(inputs_embeds=x).last_hidden_state
            x = x.permute(0, 2, 1)  # [B, 768, 129]
            x = self.fc_compress(x)  # [B, 768, 64]
            x = x.permute(0, 2, 1)  # [B, 64, 768]

        x = x.view(batch_size, 8, 8, 768)
        x = x.permute(0,3,1,2)
        outputs = self.decoder(x)

        return outputs



