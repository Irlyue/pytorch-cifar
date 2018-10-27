import torch.nn as nn

from .base import BaseNet


cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_vgg_conv_layers(vgg):
    layers = []
    in_channels = 3
    for x in vgg:
        if x == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(x))
            layers.append(nn.ReLU())
            in_channels = x
    # layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
    return nn.Sequential(*layers)


class VGG(BaseNet):
    def __init__(self, n_layers, n_classes=10):
        assert n_layers in cfg.keys(), 'Choose from %s' % cfg.keys()
        super(VGG, self).__init__(n_classes, name='VGG-%d' % n_layers)
        self.features = make_vgg_conv_layers(cfg[n_layers])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
