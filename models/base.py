import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, n_classes, name='Base'):
        super().__init__()
        self.n_classes = n_classes
        self.name = name
