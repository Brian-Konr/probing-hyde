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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, channel]
        se = self.fc1(x)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se
    

class AttentionMLPSE1DCNN(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism.
    Enhanced with SE blocks and 1DCNN for improved feature aggregation and stability.
    """
    def __init__(self, hidden_size=4096, num_layers=32, num_heads=8, dropout=0.1, reduction=16):
        super(AttentionMLPSE1DCNN, self).__init__()
        
        # multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.dropout_attention = nn.Dropout(p=dropout)
        
        # 1DCNN
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        # Residual Connection
        self.residual_conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(1)
        
        # MLP 
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.se1 = SEBlock(1024, reduction)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.se2 = SEBlock(512, reduction)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.se3 = SEBlock(256, reduction)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(256, 1)
        self.fc4_bn = nn.BatchNorm1d(1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Attention
        attn_output, attn_weights = self.attention(x, x, x)  # [32, 32, 4096]
        x = self.attention_norm(attn_output + x)           
        x = self.dropout_attention(x)
        
        # 1DCNN 
        x_main = self.conv1(x)                         # [32, 16, 4096]
        x_main = self.bn1(x_main)
        x_main = self.leaky_relu(x_main)
        
        x_main = self.conv2(x_main)                         # [32, 1, 4096]
        x_main = self.bn2(x_main)
        x_main = self.leaky_relu(x_main)
        
        # RC
        x_residual = self.residual_conv(x)              # [32, 1, 4096]
        x_residual = self.residual_bn(x_residual)
        x_cnn = x_main + x_residual                         # [32, 1, 4096]
        x_cnn = self.leaky_relu(x_cnn)
        
        # flatten
        x_cnn = x_cnn.view(x_cnn.size(0), -1)              # [32, 4096]
        
        # FC1 with SE
        fc1_out = self.fc1(x_cnn)                           # [32, 1024]
        fc1_out = self.fc1_bn(fc1_out)
        fc1_out = self.se1(fc1_out)
        fc1_out = self.relu1(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        # FC2 with SE
        fc2_out = self.fc2(fc1_out)                         # [32, 512]
        fc2_out = self.fc2_bn(fc2_out)
        fc2_out = self.se2(fc2_out)
        fc2_out = self.relu2(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        # FC3 with SE
        fc3_out = self.fc3(fc2_out)                         # [32, 256]
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = self.se3(fc3_out)
        fc3_out = self.relu3(fc3_out)
        fc3_out = self.dropout3(fc3_out)
        
        # FC4
        logits = self.fc4(fc3_out)                          # [32, 1]
        logits = self.fc4_bn(logits)
        
        return logits    
