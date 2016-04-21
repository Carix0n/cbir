import unittest
import numpy as np
from src.main.python.extra_functions import pairwise_distance

class TestExtraFunctions(unittest.TestCase):
    def test_pairwise_distance(self):
        x = np.matrix([[1, 2], [3, 4]])
        y = np.matrix([[0, 1, 2], [3, 4, 5]])
        src = pairwise_distance(x, y)
        dst = np.matrix([[1, 1, 5], [5, 1, 1]])
        self.assertEqual(src.all(), dst.all())


if __name__ == '__main__':
    unittest.main()
