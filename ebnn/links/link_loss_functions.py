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

class BitwiseXorLoss(link.Link):
    """Loss based on XOR operation"""
    def __init__(self):
        super(BitwiseXorLoss, self).__init__()
        self.cname = "l_bitwise_xor"

    def __call__(self, x, t):
        # Continuous approximation of XOR: (x-t)^2
        return F.mean_squared_error(x, t)

class BitwiseAndLoss(link.Link):
    """Loss based on AND operation: penalty if (x AND t) != t"""
    def __init__(self):
        super(BitwiseAndLoss, self).__init__()
        self.cname = "l_bitwise_and"

    def __call__(self, x, t):
        # Continuous approximation of AND: (x*t - t)^2
        return F.mean_squared_error(x * t, t)

class BitwiseOrLoss(link.Link):
    """Loss based on OR operation: penalty if (x OR t) != 1 where either is 1"""
    def __init__(self):
        super(BitwiseOrLoss, self).__init__()
        self.cname = "l_bitwise_or"

    def __call__(self, x, t):
        # Continuous approximation of OR: 1 - (1-x)*(1-t) should be close to 1 if either is 1
        # Target is x+t - x*t
        target = x + t - x * t
        return F.mean_squared_error(x, target)

class BitwiseNotLoss(link.Link):
    """Loss based on NOT operation: penalty if x != (1-t)"""
    def __init__(self):
        super(BitwiseNotLoss, self).__init__()
        self.cname = "l_bitwise_not"

    def __call__(self, x, t):
        return F.mean_squared_error(x, 1.0 - t)

class BitwiseNandLoss(link.Link):
    def __init__(self):
        super(BitwiseNandLoss, self).__init__()
        self.cname = "l_bitwise_nand"

    def __call__(self, x, t):
        return F.mean_squared_error(x, 1.0 - (x * t))

class BitwiseNorLoss(link.Link):
    def __init__(self):
        super(BitwiseNorLoss, self).__init__()
        self.cname = "l_bitwise_nor"

    def __call__(self, x, t):
        return F.mean_squared_error(x, (1.0 - x) * (1.0 - t))

class LogicalConstraintLoss(link.Link):
    """
    Enforces that the output satisfies a logical relation.
    relation can be 'implies', 'equivalent', etc.
    """
    def __init__(self, relation='implies'):
        super(LogicalConstraintLoss, self).__init__()
        self.relation = relation
        self.cname = "l_logical_constraint"

    def __call__(self, x, t):
        if self.relation == 'implies':
            # t -> x  is only violated if t is true and x is false.
            # We penalize the difference (t - x) but only when t > x.
            return F.mean(F.relu(t - F.sigmoid(x)))
        elif self.relation == 'equivalent':
            return F.mean_squared_error(F.sigmoid(x), t)
        return 0.0

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
        loss = F.huber_loss(x, t, delta=self.delta)
        return F.mean(loss)

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

class CompositeLoss(link.Link):
    """Combines multiple losses with weights"""
    def __init__(self, losses_with_weights):
        super(CompositeLoss, self).__init__()
        self.losses_with_weights = losses_with_weights
        self.cname = "l_composite_loss"

    def __call__(self, x, t):
        total_loss = 0
        for loss_func, weight in self.losses_with_weights:
            total_loss += weight * loss_func(x, t)
        return total_loss

class StateTransitionLoss(link.Link):
    """Loss for LSTM internal states to ensure smooth transitions"""
    def __init__(self, weight=0.01):
        super(StateTransitionLoss, self).__init__()
        self.weight = weight
        self.cname = "l_state_transition"

    def __call__(self, model, x, t):
        # We find any LSTM-like layers in the predictor and apply state transition loss
        total_loss = 0.0
        found = False
        for link in model.predictor:
            if hasattr(link, 'h') and hasattr(link, 'prev_h'):
                h = link.h
                prev_h = link.prev_h
                if prev_h is not None:
                    total_loss += self.weight * F.mean_squared_error(h, prev_h)
                found = True

        if not found:
            return 0.0
        return total_loss
