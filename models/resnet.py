import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(BaseNet):
    def __init__(self, block, blocks, n_classes=10):
        super().__init__(n_classes, name='ResNet')
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stage1 = self._make_layer(block, 64, blocks[0], stride=2)
        self.stage2 = self._make_layer(block, 128, blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, blocks[3], stride=1)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, n_blocks, stride):
        layers = []
        strides = [stride] + [1] * (n_blocks - 1)
        for s in strides:
            layers.append(Bottleneck(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, (out.size(2), out.size(3))).view(x.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(_ResNet):
    def __init__(self, n_layers, n_classes=10):
        assert n_layers in (18, 50, 101, 152), 'n must be choose from {18, 50, 101, 152}'
        super().__init__(Bottleneck, NETS[n_layers], n_classes=n_classes)


NETS = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 6, 3],
    152: [3, 8, 36, 3]
}
