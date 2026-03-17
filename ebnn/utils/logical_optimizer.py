import chainer
from chainer import training
from chainer.training import extension
import numpy as np

class LogicalOptimizer(chainer.Optimizer):
    """
    A wrapper optimizer that can dynamically adjust loss weights or
    apply logical constraints during training.
    Also handles weight clipping for binary networks.
    """
    def __init__(self, actual_optimizer):
        super(LogicalOptimizer, self).__init__()
        self.actual_optimizer = actual_optimizer
        self.target = actual_optimizer.target
        self.logical_weight = 0.1 # Default weight for logical components

    def update(self, loss_func=None, *args, **kwds):
        if loss_func is not None:
            self.actual_optimizer.update(loss_func, *args, **kwds)
        else:
            self.actual_optimizer.update(*args, **kwds)

        # Post-update: Weight clipping to [-1, 1]
        for param in self.target.params():
            if 'W' in param.name:
                xp = chainer.cuda.get_array_module(param.data)
                param.data[:] = xp.clip(param.data, -1.0, 1.0)

    def setup(self, link):
        self.actual_optimizer.setup(link)
        super(LogicalOptimizer, self).setup(link)

    def serialize(self, serializer):
        self.actual_optimizer.serialize(serializer)
        if 'logical_weight' in serializer:
            self.logical_weight = serializer('logical_weight', self.logical_weight)

class LogicalWeightScheduler(extension.Extension):
    def __init__(self, optimizer, attr='logical_weight', factor=0.95, interval=100):
        self.optimizer = optimizer
        self.attr = attr
        self.factor = factor
        self.interval = interval

    def __call__(self, trainer):
        if trainer.updater.iteration > 0 and trainer.updater.iteration % self.interval == 0:
            current = getattr(self.optimizer, self.attr, None)
            if current is not None:
                new_val = current * self.factor
                setattr(self.optimizer, self.attr, new_val)
