from __future__ import absolute_import

import chainer
import chainer.functions as F
import numpy as np
from .link_binary_linear import BinaryLinear
from . import CLink

class BinaryLSTM(chainer.Chain, CLink):
    def __init__(self, in_size, out_size, max_complexity=None):
        super(BinaryLSTM, self).__init__()
        self.effective_out_size = out_size
        if max_complexity is not None:
            self.effective_out_size = min(out_size, max_complexity)

        with self.init_scope():
            self.upward = BinaryLinear(in_size, 4 * self.effective_out_size)
            self.lateral = BinaryLinear(self.effective_out_size, 4 * self.effective_out_size, nobias=True)
            # Add Batch Normalization to help convergence
            self.bn = chainer.links.BatchNormalization(4 * self.effective_out_size)

            self.upward.W.data[:] = np.random.normal(0, 0.1, self.upward.W.shape).astype(np.float32)
            self.lateral.W.data[:] = np.random.normal(0, 0.1, self.lateral.W.shape).astype(np.float32)
        self.out_size = self.effective_out_size
        self.max_complexity = max_complexity
        self.cname = "l_binary_lstm"
        self.h = None
        self.c = None
        self.prev_h = None
        self.prev_c = None

    def __call__(self, x, return_state=False):
        if self.h is None:
            xp = self.xp
            self.h = chainer.Variable(xp.zeros((x.shape[0], self.out_size), dtype=x.dtype))
            self.c = chainer.Variable(xp.zeros((x.shape[0], self.out_size), dtype=x.dtype))

        self.prev_c = self.c
        self.prev_h = self.h

        # upward(x) + lateral(h) then BN
        lstm_in = self.bn(self.upward(x) + self.lateral(self.h))
        new_c, new_h = F.lstm(self.c, lstm_in)

        self.c = new_c
        self.h = new_h

        if return_state: return self.h, self.c
        return self.h

    def get_complexity(self):
        if self.h is not None:
            return F.sum(F.absolute(self.h))
        return chainer.Variable(self.xp.array(0.0, dtype=np.float32))

    def reset_state(self):
        self.h = None
        self.c = None
        self.prev_h = None
        self.prev_c = None

    def generate_c(self, link_idx, inp_shape):
        return "/* LSTM C generation not implemented */\n"
    def param_mem(self): return 0
    def temp_mem(self, inp_shape): return 0
