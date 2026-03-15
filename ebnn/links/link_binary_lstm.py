from __future__ import absolute_import

import chainer
import chainer.functions as F
import numpy as np
from .link_binary_linear import BinaryLinear
from . import CLink

class BinaryLSTM(chainer.Chain, CLink):
    def __init__(self, in_size, out_size, max_complexity=None):
        super(BinaryLSTM, self).__init__()
        # If max_complexity is set, we limit the effective out_size
        self.effective_out_size = out_size
        if max_complexity is not None:
            self.effective_out_size = min(out_size, max_complexity)

        with self.init_scope():
            self.upward = BinaryLinear(in_size, 4 * self.effective_out_size)
            self.lateral = BinaryLinear(self.effective_out_size, 4 * self.effective_out_size, nobias=True)
        self.out_size = self.effective_out_size
        self.max_complexity = max_complexity
        self.cname = "l_binary_lstm"
        self.h = None
        self.c = None

    def __call__(self, x):
        if self.h is None:
            xp = self.xp
            self.h = chainer.Variable(xp.zeros((x.shape[0], self.out_size), dtype=x.dtype))
            self.c = chainer.Variable(xp.zeros((x.shape[0], self.out_size), dtype=x.dtype))

        lstm_in = self.upward(x) + self.lateral(self.h)
        self.c, self.h = F.lstm(self.c, lstm_in)

        return self.h

    def reset_state(self):
        self.h = None
        self.c = None

    def generate_c(self, link_idx, inp_shape):
        name = self.cname + str(link_idx)
        return "/* LSTM C generation for {} not fully implemented in this version */\n".format(name)

    def param_mem(self):
        # Rough estimate of parameter memory in bytes
        # upward: (in_size * 4 * out_size) / 8 (binary weights) + 4 * out_size * 4 (float bias)
        # lateral: (out_size * 4 * out_size) / 8 (binary weights)
        return (4 * self.out_size * (1 + self.out_size // 8)) + 16 * self.out_size

    def temp_mem(self, inp_shape):
        # Temporary memory for intermediate activations
        return 4 * self.out_size * 4
