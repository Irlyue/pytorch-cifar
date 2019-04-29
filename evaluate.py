import torch
import engine

import models
import inputs
import utility

cf = utility.load_config_from_environ()

net = models.load_net(cf['network'], cf['n_classes'])
g = engine.Engine(net, cf)
engine.setup_logger(g.absolute_path(engine.EVAL_LOG_FILE))

_, eval_loader = inputs.load_cifar10(cf['batch_size'], cf['device'])
metrics = [
    engine.AccuracyMetric(func=lambda _x, _y: (_x.to(torch.long), _y.argmax(-1))),
]
g.evaluate(eval_loader, metrics)
