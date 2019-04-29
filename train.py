import torch
import engine
import torch.nn as nn
import torch.optim as optim

import models
import inputs
import utility


config = utility.load_config_from_environ()

net = models.load_net(config['network'], config['n_classes'])
g = engine.Engine(net, config)

engine.setup_logger(g.absolute_path(engine.TRAIN_LOG_FILE))
engine.setup_visdom(port=1091)

optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])
loss_fn = nn.CrossEntropyLoss()
train_loader, eval_loader = inputs.load_cifar10(config['batch_size'], config['device'])
metrics = [
    engine.AccuracyMetric(func=lambda _x, _y: (_x.to(torch.long), _y.argmax(-1))),
]
hooks = [
    engine.ScalarSummaryHook([{'x': 'epoch', 'y': 'loss'}], summary_every_steps=100),
    engine.SaveBestModelHook(metrics[0], eval_loader),
    engine.SaveModelHook(save_every_steps=int(1e7)),
]

g.train(train_loader, optimizer, loss_fn, hooks=hooks, progress_bar=False)
