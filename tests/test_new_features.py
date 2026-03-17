import unittest
import numpy as np
import chainer
import os
import sys

# Setup path to import ebnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ebnn.links as BL

class TestLossFunctions(unittest.TestCase):
    def test_mse(self):
        loss = BL.MeanSquaredError()
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        t = np.array([[1.5, 1.5]], dtype=np.float32)
        l = loss(x, t)
        self.assertAlmostEqual(l.data, 0.25)

    def test_bitwise_xor(self):
        loss = BL.BitwiseXorLoss()
        # XOR is 1 when inputs are different. Loss = 1 - XOR.
        # So XOR loss is 0 when inputs are different.
        x = np.array([[100.0, -100.0]], dtype=np.float32) # [1, 0]
        t = np.array([[0.0, 1.0]], dtype=np.float32)      # [0, 1]
        l = loss(x, t)
        self.assertAlmostEqual(l.data, 0.0, places=5)

    def test_logical_constraint_equivalent(self):
        loss = BL.LogicalConstraintLoss('equivalent')
        x = np.array([[100.0]], dtype=np.float32) # Sigmoid(100) approx 1
        t = np.array([[1.0]], dtype=np.float32)
        l = loss(x, t)
        self.assertAlmostEqual(l.data, 0.0, places=5)

    def test_logical_constraint_implies(self):
        loss = BL.LogicalConstraintLoss('implies')
        # t -> x
        # 1 -> 0 should have loss
        x = np.array([[-100.0]], dtype=np.float32) # Sigmoid is 0
        t = np.array([[1.0]], dtype=np.float32)
        l = loss(x, t)
        self.assertGreater(l.data, 0.5)

        # 0 -> 1 should have no loss
        x = np.array([[100.0]], dtype=np.float32) # Sigmoid is 1
        t = np.array([[0.0]], dtype=np.float32)
        l = loss(x, t)
        self.assertAlmostEqual(l.data, 0.0)

        # 1 -> 1 should have no loss
        x = np.array([[100.0]], dtype=np.float32)
        t = np.array([[1.0]], dtype=np.float32)
        l = loss(x, t)
        self.assertAlmostEqual(l.data, 0.0)

    def test_composite_loss(self):
        mse = BL.MeanSquaredError()
        abs_err = BL.AbsoluteError()
        comp = BL.CompositeLoss([(mse, 1.0), (abs_err, 0.5)])
        x = np.array([[1.0]], dtype=np.float32)
        t = np.array([[2.0]], dtype=np.float32)
        # MSE = (1-2)^2 = 1
        # ABS = |1-2| = 1
        # Total = 1*1 + 0.5*1 = 1.5
        l = comp(x, t)
        self.assertEqual(l.data, 1.5)

class TestLSTM(unittest.TestCase):
    def test_lstm_complexity(self):
        # Test max_complexity constraint
        lstm = BL.BinaryLSTM(1, 32, max_complexity=8)
        self.assertEqual(lstm.out_size, 8)
        self.assertEqual(lstm.upward.W.shape[0], 4 * 8)

    def test_lstm_state_transition_loss(self):
        lstm = BL.BinaryLSTM(1, 10)
        class MockModel:
            def __init__(self, predictor):
                self.predictor = chainer.Sequential(predictor)

        model = MockModel(lstm)
        loss_func = BL.StateTransitionLoss(weight=1.0)

        x = np.zeros((1, 1), dtype=np.float32)
        lstm(x) # Set states
        l = loss_func(model, x, x)
        self.assertEqual(l.data, 0.0) # prev_h is None or same as h initially?
        # Actually in __call__ we set prev_h = h BEFORE updating h?
        # Wait, let's check link_binary_lstm.py

if __name__ == '__main__':
    unittest.main()
