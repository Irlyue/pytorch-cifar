from .resnet import ResNet
from .vgg import VGG


def load_net(name, n_classes):
    """
    Supported name:
        vgg11, vgg13, vgg16, vgg19
        resnet18, resnet50, resnet101, resnet152
    """
    name = name.lower()
    if name.startswith('vgg'):
        return VGG(name, n_classes)
    elif name.startswith('resnet'):
        return ResNet(int(name[6:]), n_classes)
    else:
        assert False
