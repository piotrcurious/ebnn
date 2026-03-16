import chainer
import numpy as np

class StructuralAwareness(object):
    """
    Introspects model structure and connectivity to provide feedback for optimization.
    """
    @staticmethod
    def map_connectivity(model):
        """
        Maps the strength and sparsity of connections across layers.
        """
        map_info = {}
        for name, param in model.namedparams():
            if 'W' in name:
                W = param.data
                xp = chainer.cuda.get_array_module(W)
                # Binary networks have weights close to -1 or 1
                # Awareness of "undecided" weights (near 0)
                undecided = xp.mean(xp.abs(W) < 0.2)
                map_info[name] = {
                    'shape': W.shape,
                    'undecided_ratio': float(undecided),
                    'l1_norm': float(xp.sum(xp.abs(W))),
                    'entropy': float(-xp.mean(xp.abs(W) * xp.log(xp.abs(W) + 1e-7)))
                }
        return map_info

    @staticmethod
    def get_path_bottlenecks(model, activations):
        """
        Identifies layers or neurons that are highly saturated or inactive.
        """
        bottlenecks = {}
        for name, act in activations.items():
            # In binary logic, we want values to be close to 0 or 1
            # Bottleneck if many values are stuck at 0.5
            variance = chainer.functions.mean((act - 0.5)**2).data
            bottlenecks[name] = float(variance)
        return bottlenecks
