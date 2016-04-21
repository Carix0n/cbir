import os
import numpy as np
from numpy import matlib


def list_image_dir(image_base_path):
    files = os.listdir(image_base_path)
    if files[0] == 'Thumbs.db':
        files = files[1:]
    files = [elem for elem in files if os.path.isfile(os.path.join(image_base_path, elem))]
    return files


def pairwise_distance(x, y):
    assert (type(x) is np.ndarray or type(x) is np.matrix) and x.ndim == 2
    assert (type(y) is np.ndarray or type(y) is np.matrix) and y.ndim == 2
    assert x.shape[0] == y.shape[0]
    # (x - y, x - y) = (x, x) - 2(x, y) + (y, y)
    x = np.transpose(x)
    x_norm = np.matlib.repmat(np.matrix(np.square(x)).sum(axis=1), 1, y.shape[1])
    y_norm = np.matlib.repmat(np.matrix(np.square(y)).sum(axis=0), x.shape[1], 1)
    return x_norm - 2 * np.dot(x, y) + y_norm