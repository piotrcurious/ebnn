from __future__ import absolute_import

from chainer import link
import chainer.functions as F

class MeanSquaredError(link.Link):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.cname = "l_mse"

    def __call__(self, x, t):
        return F.mean_squared_error(x, t)

class AbsoluteError(link.Link):
    def __init__(self):
        super(AbsoluteError, self).__init__()
        self.cname = "l_absolute_error"

    def __call__(self, x, t):
        return F.mean_absolute_error(x, t)

class BinaryError(link.Link):
    """Loss for binary classification"""
    def __init__(self):
        super(BinaryError, self).__init__()
        self.cname = "l_binary_error"

    def __call__(self, x, t):
        return F.sigmoid_cross_entropy(x, t)

class HammingLoss(link.Link):
    """Bitwise loss"""
    def __init__(self):
        super(HammingLoss, self).__init__()
        self.cname = "l_hamming_loss"

    def __call__(self, x, t):
        return F.mean(F.absolute_error(F.sigmoid(x), t))

class HingeLoss(link.Link):
    """Algebraic/Binary loss"""
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.cname = "l_hinge_loss"

    def __call__(self, x, t):
        return F.hinge(x, t)

class HuberLoss(link.Link):
    """Algebraic loss"""
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.cname = "l_huber_loss"

    def __call__(self, x, t):
        return F.huber_loss(x, t, delta=self.delta)

class CrossEntropyLoss(link.Link):
    """Algebraic/Binary loss"""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cname = "l_cross_entropy"

    def __call__(self, x, t):
        return F.softmax_cross_entropy(x, t)

class ComplexityRegularizer(link.Link):
    """Adds a penalty based on model complexity (e.g. number of active units)"""
    def __init__(self, base_loss, weight=0.01):
        super(ComplexityRegularizer, self).__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.cname = "l_complexity_regularizer"

    def __call__(self, y, t):
        loss = self.base_loss(y, t)
        # Complexity as L1 norm of activations
        complexity = F.sum(F.absolute(y))
        return loss + self.weight * complexity
