import chainer
import chainer.functions as F
import numpy as np

class SymbolicEnergy(object):
    """
    Base class for symbolic energy components that can be composed.
    Each component represents a term in the Hamiltonian H = sum(alpha_i * E_i).
    """
    def __init__(self, weight=1.0, name=None):
        self.weight = weight
        self.name = name or self.__class__.__name__

    def compute(self, model, x, y, t):
        raise NotImplementedError()

    def __add__(self, other):
        return EnergyComposition(self, other, '+')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            self.weight *= other
            return self
        return EnergyComposition(self, other, '*')

    def __rmul__(self, other):
        return self.__mul__(other)

class EnergyComposition(SymbolicEnergy):
    def __init__(self, e1, e2, op):
        super(EnergyComposition, self).__init__()
        self.e1 = e1
        self.e2 = e2
        self.op = op
        self.name = f"({e1.name} {op} {e2.name})"

    def compute(self, model, x, y, t):
        v1 = self.e1.compute(model, x, y, t)
        v2 = self.e2.compute(model, x, y, t)
        if self.op == '+': return v1 + v2
        if self.op == '*': return v1 * v2
        return v1

class PotentialEnergy(SymbolicEnergy):
    """Represents the data-driven error (Potential)."""
    def compute(self, model, x, y, t):
        return F.mean_squared_error(y, t)

class InteractionEnergy(SymbolicEnergy):
    """Represents the interaction between model weights (Ising-like)."""
    def compute(self, model, x, y, t):
        energy = 0
        for param in model.params():
            if 'W' in param.name:
                # Local interaction: negative product of neighbors
                # H_int = - sum W_i * W_j
                if param.ndim == 2:
                    energy -= F.sum(param[:, :-1] * param[:, 1:])
                    energy -= F.sum(param[:-1, :] * param[1:, :])
        return self.weight * energy

class KineticEnergy(SymbolicEnergy):
    """Represents the energy of state transitions (for RNN/LSTM)."""
    def compute(self, model, x, y, t):
        energy = 0
        for link in model.children():
            if hasattr(link, 'h') and hasattr(link, 'prev_h'):
                if link.prev_h is not None:
                    energy += F.mean_squared_error(link.h, link.prev_h)
        return self.weight * energy

class LogicalConstraintEnergy(SymbolicEnergy):
    """Represents the energy of violating logical predicates."""
    def __init__(self, predicate_func, weight=1.0, name=None):
        super(LogicalConstraintEnergy, self).__init__(weight, name)
        self.predicate_func = predicate_func

    def compute(self, model, x, y, t):
        # We assume predicate_func returns a value in [0, 1] where 1 is satisfied
        # Energy is 1 - satisfaction
        sat = self.predicate_func(y, t)
        return self.weight * F.mean(1.0 - sat)

class HamiltonianSystem(chainer.Link):
    """
    The main coordinator for the Energy-Based Loss system.
    """
    def __init__(self, energy_expression):
        super(HamiltonianSystem, self).__init__()
        self.energy_expression = energy_expression

    def __call__(self, model, x, y, t):
        total_energy = self.energy_expression.compute(model, x, y, t)
        return total_energy
