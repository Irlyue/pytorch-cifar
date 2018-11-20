import json
import torch
import engine
import torch.nn.functional as F

import models
import inputs
import utility

config = utility.load_config_from_environ()
print('\n%s\n' % json.dumps(config, indent=2))

net = models.load_net(config['network'], config['n_classes'])
g = engine.Engine(net, config)
_, eval_loader = inputs.load_cifar10(config['batch_size'], config['device'])
metrics = [
    engine.AccuracyMetric(func=lambda _x, _y: (_x.to(torch.long), _y.argmax(-1))),
    engine.CrossEntropyLossMetric(func=lambda _x, _y: (_x.to(torch.long), F.softmax(_y, -1))),
]
g.evaluate(eval_loader, metrics)
