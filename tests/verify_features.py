
import os
import sys
import numpy as np
import chainer
import chainer.functions as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ebnn.links as BL

def test_losses():
    x = chainer.Variable(np.array([[0.5], [-0.5]], dtype=np.float32))
    t_f = chainer.Variable(np.array([[1.0], [0.0]], dtype=np.float32))
    t_i = chainer.Variable(np.array([1, 0], dtype=np.int32))

    for l in [BL.MeanSquaredError(), BL.AbsoluteError(), BL.HuberLoss()]:
        loss = l(x, t_f)
        print(f"{l.cname}: {loss.data}")
        assert loss.data >= 0

    # BinaryError expects same shape if float, or specific shape if int
    l = BL.BinaryError()
    loss = l(x, t_f) # Using float targets of same shape
    print(f"{l.cname}: {loss.data}")
    assert loss.data >= 0

    # Bitwise and logical losses
    for l in [BL.HammingLoss(), BL.BitwiseXorLoss(), BL.BitwiseAndLoss(),
              BL.BitwiseOrLoss(), BL.BitwiseNotLoss(), BL.BitwiseNandLoss(),
              BL.BitwiseNorLoss(), BL.LogicalConstraintLoss('implies'),
              BL.LogicalConstraintLoss('equivalent')]:
        loss = l(x, t_f)
        print(f"{l.cname}: {loss.data}")
        assert loss.data >= 0

def test_lstm_complexity():
    lstm = BL.BinaryLSTM(1, 64, max_complexity=16)
    assert lstm.out_size == 16
    x = chainer.Variable(np.random.randn(5, 1).astype(np.float32))
    y = lstm(x)
    assert y.shape == (5, 16)
    print("LSTM max_complexity test passed")

if __name__ == "__main__":
    test_losses()
    test_lstm_complexity()
