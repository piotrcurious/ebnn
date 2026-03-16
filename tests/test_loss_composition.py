import unittest
import numpy as np
import chainer
import os
import sys

# Setup path to import ebnn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ebnn.links as BL

class TestLossComposition(unittest.TestCase):
    def test_add(self):
        l1 = BL.MeanSquaredError()
        l2 = BL.AbsoluteError()
        l_sum = l1 + l2

        x = np.array([[1.0]], dtype=np.float32)
        t = np.array([[2.0]], dtype=np.float32)
        # MSE = (1-2)^2 = 1
        # ABS = |1-2| = 1
        # Sum = 2
        res = l_sum(x, t)
        self.assertEqual(res.data, 2.0)

    def test_mul_constant(self):
        l1 = BL.MeanSquaredError()
        l_mul = l1 * 0.5

        x = np.array([[1.0]], dtype=np.float32)
        t = np.array([[3.0]], dtype=np.float32)
        # MSE = (1-3)^2 = 4
        # Mul = 4 * 0.5 = 2
        res = l_mul(x, t)
        self.assertEqual(res.data, 2.0)

    def test_complex_composition(self):
        l1 = BL.MeanSquaredError()
        l2 = BL.AbsoluteError()
        l_comp = 0.7 * l1 + 0.3 * l2 + 5.0

        x = np.array([[0.0]], dtype=np.float32)
        t = np.array([[1.0]], dtype=np.float32)
        # MSE = 1, ABS = 1
        # 0.7*1 + 0.3*1 + 5.0 = 6.0
        res = l_comp(x, t)
        self.assertAlmostEqual(res.data, 6.0)

if __name__ == '__main__':
    unittest.main()
