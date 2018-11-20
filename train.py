import os
import json
import torch
import engine
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import models
import inputs
import utility

config = utility.load_config_from_environ()
print('\n%s\n' % json.dumps(config, indent=2))


net = models.load_net(config['network'], config['n_classes'])
g = engine.Engine(net, config)
optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])
loss_fn = nn.CrossEntropyLoss()
train_loader, eval_loader = inputs.load_cifar10(config['batch_size'], config['device'])
metrics = [
    engine.AccuracyMetric(func=lambda _x, _y: (_x.to(torch.long), _y.argmax(-1))),
    engine.CrossEntropyLossMetric(func=lambda _x, _y: (_x.to(torch.long), F.softmax(_y, -1))),
]
hooks = [
    # engine.ExponentialMovingAverageHook(net.parameters(), device=config['device']),
    engine.LoggingHook(['step', 'loss'], print_every_epochs=1),
    engine.SaveModelHook(save_every_epochs=1),
    engine.SaveBestModelHook(metrics[0], eval_loader)
]

g.train(train_loader, optimizer, loss_fn, hooks=hooks)
