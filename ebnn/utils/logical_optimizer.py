import chainer
from chainer import training
from chainer.training import extension

class LogicalOptimizer(chainer.Optimizer):
    """
    A wrapper optimizer that can dynamically adjust loss weights or
    apply logical constraints during training.
    """
    def __init__(self, actual_optimizer):
        super(LogicalOptimizer, self).__init__()
        self.actual_optimizer = actual_optimizer
        self.target = actual_optimizer.target
        self.logical_weight = 0.1 # Default weight for logical components

    def update(self, loss_func=None, *args, **kwds):
        """
        Updates parameters.
        If loss_func is provided, it is used to compute the loss.
        """
        if loss_func is not None:
            # We can't easily wrap loss_func here without knowing its signature
            # but we can let the actual optimizer handle it.
            self.actual_optimizer.update(loss_func, *args, **kwds)
        else:
            self.actual_optimizer.update(*args, **kwds)

    def setup(self, link):
        self.actual_optimizer.setup(link)
        super(LogicalOptimizer, self).setup(link)

    def serialize(self, serializer):
        self.actual_optimizer.serialize(serializer)
        if 'logical_weight' in serializer:
            self.logical_weight = serializer('logical_weight', self.logical_weight)

class LogicalWeightScheduler(extension.Extension):
    """
    Extension to adjust the weight of logical losses over time.
    Useful for 'warm-starting' with logical constraints and then
    decaying them to focus on precision, or vice versa.
    """
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
