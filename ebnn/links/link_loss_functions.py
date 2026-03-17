from __future__ import absolute_import

from chainer import link
import chainer.functions as F
import chainer
from ebnn.utils import logic as L
from ebnn.utils.hamiltonian import Hamiltonian

class BaseLoss(link.Link):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.cname = "l_base"

    def __call__(self, *args, **kwargs):
        # We handle (x, t) or (model, x, t)
        if len(args) == 3:
            return self.forward(args[1], args[2], model=args[0])
        return self.forward(args[0], args[1])

    def forward(self, x, t, model=None):
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, (int, float)): return AddConstantLoss(self, other)
        return AddLoss(self, other)

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)): return AddConstantLoss(self, -other)
        return SubLoss(self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)): return MulConstantLoss(self, other)
        return MulLoss(self, other)

    def __rmul__(self, other): return self.__mul__(other)

class AddLoss(BaseLoss):
    def __init__(self, l1, l2):
        super(AddLoss, self).__init__()
        with self.init_scope():
            self.l1 = l1
            self.l2 = l2
        self.cname = f"({l1.cname}+{l2.cname})"
    def __call__(self, *args, **kwargs): return self.l1(*args, **kwargs) + self.l2(*args, **kwargs)

class SubLoss(BaseLoss):
    def __init__(self, l1, l2):
        super(SubLoss, self).__init__()
        with self.init_scope():
            self.l1 = l1
            self.l2 = l2
        self.cname = f"({l1.cname}-{l2.cname})"
    def __call__(self, *args, **kwargs): return self.l1(*args, **kwargs) - self.l2(*args, **kwargs)

class MulLoss(BaseLoss):
    def __init__(self, l1, l2):
        super(MulLoss, self).__init__()
        with self.init_scope():
            self.l1 = l1
            self.l2 = l2
        self.cname = f"({l1.cname}*{l2.cname})"
    def __call__(self, *args, **kwargs): return self.l1(*args, **kwargs) * self.l2(*args, **kwargs)

class AddConstantLoss(BaseLoss):
    def __init__(self, l, c):
        super(AddConstantLoss, self).__init__()
        with self.init_scope(): self.l = l
        self.c = c
        self.cname = f"({l.cname}+{c})"
    def __call__(self, *args, **kwargs): return self.l(*args, **kwargs) + self.c

class MulConstantLoss(BaseLoss):
    def __init__(self, l, c):
        super(MulConstantLoss, self).__init__()
        with self.init_scope(): self.l = l
        self.c = c
        self.cname = f"({l.cname}*{c})"
    def __call__(self, *args, **kwargs): return self.l(*args, **kwargs) * self.c

class MeanSquaredError(BaseLoss):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.cname = "l_mse"
    def forward(self, x, t, model=None): return F.mean_squared_error(x, t)

class AbsoluteError(BaseLoss):
    def __init__(self):
        super(AbsoluteError, self).__init__()
        self.cname = "l_absolute_error"
    def forward(self, x, t, model=None): return F.mean_absolute_error(x, t)

class BinaryError(BaseLoss):
    def __init__(self):
        super(BinaryError, self).__init__()
        self.cname = "l_binary_error"
    def forward(self, x, t, model=None):
        if x.shape != t.shape: t = F.reshape(t, x.shape)
        if t.dtype.kind == 'f': t = (t > 0.5).astype('int32')
        return F.sigmoid_cross_entropy(x, t)

class LogicLoss(BaseLoss):
    def __init__(self, relation='equivalent', activation='sigmoid'):
        super(LogicLoss, self).__init__()
        self.relation = relation
        self.activation = activation
        self.cname = f"l_logic_{relation}"

    def forward(self, x, t, model=None):
        if x.shape != t.shape: t = F.reshape(t, x.shape)
        if self.activation == 'sigmoid':
            p_x = F.sigmoid(x)
            p_t = t
        elif self.activation == 'tanh':
            p_x = (F.tanh(x) + 1.0) / 2.0
            p_t = (t + 1.0) / 2.0
        else:
            p_x = x
            p_t = t

        if self.relation == 'equivalent': val = L.f_equivalent(p_x, p_t)
        elif self.relation == 'implies': val = L.f_implies(p_t, p_x)
        elif self.relation == 'and': val = L.f_and(p_x, p_t)
        elif self.relation == 'or': val = L.f_or(p_x, p_t)
        elif self.relation == 'xor': val = L.f_xor(p_x, p_t)
        elif self.relation == 'nand': val = L.f_nand(p_x, p_t)
        elif self.relation == 'nor': val = L.f_nor(p_x, p_t)
        elif self.relation == 'not': val = L.f_not(p_x)
        else: val = p_x

        return F.mean(1.0 - val)

LogicalConstraintLoss = LogicLoss
BitwiseXorLoss = lambda: LogicLoss('xor')
BitwiseAndLoss = lambda: LogicLoss('and')
BitwiseOrLoss = lambda: LogicLoss('or')
BitwiseNotLoss = lambda: LogicLoss('not')
BitwiseNandLoss = lambda: LogicLoss('nand')
BitwiseNorLoss = lambda: LogicLoss('nor')
HammingLoss = lambda: LogicLoss('equivalent')

class HuberLoss(BaseLoss):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.cname = "l_huber"
    def forward(self, x, t, model=None): return F.mean(F.huber_loss(x, t, delta=self.delta))

class HingeLoss(BaseLoss):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.cname = "l_hinge"
    def forward(self, x, t, model=None):
        if t.dtype.kind == 'f': t = (t > 0.5).astype('int32')
        return F.hinge(x, t)

class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cname = "l_cross_entropy"
    def forward(self, x, t, model=None): return F.softmax_cross_entropy(x, t)

class ComplexityLoss(BaseLoss):
    def __init__(self, weight=0.01):
        super(ComplexityLoss, self).__init__()
        self.weight = weight
        self.cname = "l_complexity"

    def forward(self, x, t, model=None):
        total_complexity = 0
        found = False
        if model is not None and hasattr(model, 'predictor'):
            for link in model.predictor:
                if hasattr(link, 'get_complexity'):
                    total_complexity += link.get_complexity()
                    found = True

        if not found:
            total_complexity = F.sum(F.absolute(x))

        return self.weight * total_complexity

class StateTransitionLoss(BaseLoss):
    def __init__(self, weight=1.0):
        super(StateTransitionLoss, self).__init__()
        self.weight = weight
        self.cname = "l_state_transition"

    def forward(self, x, t, model=None):
        total_loss = 0.0
        found = False
        if model is not None and hasattr(model, 'predictor'):
            for link in model.predictor:
                if hasattr(link, 'h') and hasattr(link, 'prev_h'):
                    if link.prev_h is not None:
                        total_loss += F.mean_squared_error(link.h, link.prev_h)
                    found = True
        if not found: return 0.0
        return self.weight * total_loss

class HamiltonianEnergyLoss(BaseLoss):
    def __init__(self, weight=0.01, mode='ising'):
        super(HamiltonianEnergyLoss, self).__init__()
        self.weight = weight
        self.mode = mode
        self.cname = f"l_hamiltonian_{mode}"

    def forward(self, x, t, model=None):
        energy = 0
        if model is None: return 0.0

        if self.mode == 'ising':
            for param in model.params():
                if 'W' in param.name:
                    energy += Hamiltonian.ising_energy(param)
        elif self.mode == 'activation':
            # Activation energy of the current output
            energy = Hamiltonian.activation_energy(x)

        return self.weight * energy

class CompositeLoss(BaseLoss):
    def __init__(self, losses_with_weights):
        super(CompositeLoss, self).__init__()
        self.losses_with_weights = losses_with_weights
        self.cname = "l_composite"

    def __call__(self, *args, **kwargs):
        total = 0
        for l, w in self.losses_with_weights:
            total += w * l(*args, **kwargs)
        return total

ComplexityRegularizer = lambda base_loss, weight=0.01: base_loss + ComplexityLoss(weight)

class LossBuilder(object):
    _loss_map = {
        'mse': MeanSquaredError,
        'abs': AbsoluteError,
        'binary': BinaryError,
        'huber': HuberLoss,
        'hinge': HingeLoss,
        'ce': CrossEntropyLoss,
        'xor': lambda: LogicLoss('xor'),
        'and': lambda: LogicLoss('and'),
        'or': lambda: LogicLoss('or'),
        'not': lambda: LogicLoss('not'),
        'implies': lambda: LogicLoss('implies'),
        'equiv': lambda: LogicLoss('equivalent'),
        'ising': lambda: HamiltonianEnergyLoss(mode='ising'),
        'energy': lambda: HamiltonianEnergyLoss(mode='activation'),
    }

    @classmethod
    def build(cls, expression):
        safe_names = {k: v() for k, v in cls._loss_map.items()}
        safe_names['complexity'] = ComplexityLoss()
        safe_names['state_transition'] = StateTransitionLoss()
        safe_names['hamiltonian'] = HamiltonianEnergyLoss()
        try:
            return eval(expression, {"__builtins__": None}, safe_names)
        except Exception as e:
            raise ValueError(f"Could not build loss from expression '{expression}': {e}")
