#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:11:47 2024

@author: paveenhuang
"""

import torch.nn as nn
    
    
class SAPLMAClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SAPLMAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


class AttentionMLP(nn.Module):
    """
    attention 
    """
    def __init__(self, input_dim=4096):
        super(AttentionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, x):
        """
        """
        x = self.relu(self.fc1(x))

        # 将输入通过注意力层
        x = x.unsqueeze(0)  # 为适配注意力机制，将 x 的维度变为 (1, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)  # 去掉多余的维度

        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.sigmoid(self.out(x))