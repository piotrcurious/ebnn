import chainer
import chainer.functions as F
import numpy as np

class Hamiltonian(object):
    """
    Utility for defining and computing Hamiltonian energies for BNNs.
    Can represent Ising-like energies or general activation-based energies.
    """
    @staticmethod
    def ising_energy(weights):
        """
        Computes the Ising energy of a weight matrix: H = -sum(W_ij * W_jk)
        This encourages alignment/stability in weights.
        """
        # Simple local interaction energy
        xp = chainer.cuda.get_array_module(weights.data)
        if weights.ndim == 2:
            # Interaction between neighboring weights
            return -F.sum(weights[:, :-1] * weights[:, 1:]) - F.sum(weights[:-1, :] * weights[1:, :])
        return chainer.Variable(xp.array(0.0, dtype=np.float32))

    @staticmethod
    def activation_energy(h):
        """
        Computes energy based on activation states.
        For binary systems, could favor certain bit patterns.
        """
        # Encourage sparsity or specific transition energies
        return F.sum(F.square(h))

class HamiltonianLoss(chainer.link.Link):
    def __init__(self, model, weight=0.01):
        super(HamiltonianLoss, self).__init__()
        self.model = model
        self.weight = weight

    def __call__(self, x=None, t=None):
        total_h = 0
        for param in self.model.params():
            if 'W' in param.name:
                total_h += Hamiltonian.ising_energy(param)
        return self.weight * total_h
