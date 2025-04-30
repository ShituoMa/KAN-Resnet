import torch
from torch import nn
import sys

import torch
import torch.nn as nn
from components.KANConv import KAN_Convolutional_Layer
from components.KANLinear import KANLinear  # 确保路径正确

class HybridBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 普通卷积层替换KAN卷积
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)

class KANResNet18(nn.Module):
    def __init__(self, grid_size=3, num_classes=7):
        super().__init__()
        # 首层保持KAN卷积
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1,
            n_convs=4,
            kernel_size=(3,3),
            stride=(1,1),
            grid_size=grid_size,
            out_channels=4,
            padding=(1,1)
        )
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

        # 中间使用普通ResNet块
        self.layer1 = self._make_layer(4, 8, blocks=1)
        self.layer2 = self._make_layer(8, 16, blocks=1, stride=2)
        self.layer3 = self._make_layer(16, 32, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 双KANLinear层结构
        self.kan_fc = nn.Sequential(
            KANLinear(32, 128, grid_size=grid_size),
            nn.ReLU(),
            KANLinear(128, num_classes, grid_size=grid_size)
        )
        self.name = 'KANResNet-18'

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [HybridBasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(HybridBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.kan_fc(x)
        return x

def create_model(device, batch_size=8):
    torch.cuda.empty_cache()
    model = HybridKANResNet18(grid_size=3, num_classes=7).to(device)
    return model