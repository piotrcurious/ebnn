import os
import sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

# Setup path to import ebnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ebnn.links as BL

def test_loss_system():
    mse = BL.MeanSquaredError()
    abs_err = BL.AbsoluteError()
    xor_loss = BL.LogicLoss('xor')

    # Algebraic composition
    combined = (mse + abs_err) * 0.5 + xor_loss * 0.1

    print(f"Combined loss name: {combined.cname}")

    x = chainer.Variable(np.array([[0.5]], dtype=np.float32))
    t = chainer.Variable(np.array([[1.0]], dtype=np.float32))

    loss_val = combined(x, t)
    print(f"Loss value: {loss_val.data}")

    # Verify it's differentiable
    loss_val.backward()
    print("Backward pass successful")

if __name__ == '__main__':
    test_loss_system()
