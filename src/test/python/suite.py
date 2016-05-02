import unittest
import numpy as np
from src.main.python.neural_codes import nc_distances
from src.main.python.extra_functions import pairwise_distance


class TestExtraFunctions(unittest.TestCase):
    def test_pairwise_distance_1(self):
        x = np.matrix([[1, 3], [2, 4]])
        y = np.matrix([[0, 3], [1, 4], [2, 5]])

        src = pairwise_distance(x, y)
        dst = np.matrix([[1, 1, 5], [5, 1, 1]])

        np.testing.assert_array_equal(src, dst)

    def test_pairwise_distance_2(self):
        x = np.matrix([[1, 2, 4], [2, 8, -1]])
        y = np.matrix([[-5, -3, 9], [1, 0, 3], [2, 8, 5], [3, 2, -2]])

        src = pairwise_distance(x, y)
        dst = np.matrix([[86, 5, 38, 40], [270, 81, 36, 38]])

        np.testing.assert_array_equal(src, dst)

    def test_nc_distances(self):
        dist_matrix = np.array([[0.2, 1.4, 2.2, 4.1, 3.2, 1.1],
                                [3.4, 2.8, 1.9, 0.1, 2.4, 5.7],
                                [0.3, 3.3, 0.1, 2.8, 3.5, 6.8],
                                [8.2, 2.5, 3.7, 6.4, 2.7, 1.2],
                                [3.5, 3.6, 3.8, 6.5, 4.1, 5.2]],
                               dtype=np.float32)

        src = nc_distances(dist_matrix, (5, 2, 3))
        dst = np.array([1.64, 1.86], dtype=np.float32)

        np.testing.assert_array_almost_equal(src, dst)


if __name__ == '__main__':
    unittest.main()
