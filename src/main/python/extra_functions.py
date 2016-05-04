import os
import numpy as np
from numpy import matlib
import skimage
from skimage import io
import matplotlib.pyplot as plt


def list_image_dir(imbase_path):
    files = os.listdir(imbase_path)

    if files[0] == 'Thumbs.db':
        files = files[1:]

    files = [os.path.join(imbase_path, elem) for elem in files if os.path.isfile(os.path.join(imbase_path, elem))]
    return files


def gen_file_name(prefix, ext, *parts):
    return os.path.join(prefix, '_'.join(parts) + ext)


def pairwise_distance(x, y):
    # vectors are in rows
    assert (type(x) is np.ndarray or type(x) is np.matrix) and x.ndim == 2
    assert (type(y) is np.ndarray or type(y) is np.matrix) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    # (x - y, x - y) = (x, x) - 2(x, y) + (y, y)
    y = np.transpose(y)
    x_norm = np.matlib.repmat(np.matrix(np.square(x)).sum(axis=1), 1, y.shape[1])
    y_norm = np.matlib.repmat(np.matrix(np.square(y)).sum(axis=0), x.shape[0], 1)
    return x_norm - 2 * np.dot(x, y) + y_norm


def pca_matrix(x):
    assert (type(x) is np.ndarray and x.ndim == 2) or type(x) is np.matrix

    x -= np.mean(x, axis=0)
    cov = np.dot(np.transpose(x), x) / (x.shape[0] - 1)
    u, s, v = np.linalg.svd(cov)
    return u


def pca_apply(x, u_reduce, n_components):
    assert (type(x) is np.ndarray and x.ndim == 2) or type(x) is np.matrix
    assert (type(u_reduce) is np.ndarray and u_reduce.ndim == 2) or type(u_reduce) is np.matrix
    assert type(n_components) is int

    return np.dot(x, u_reduce[:, :n_components])


def show_nearest(input_image, imbase, sorted_indices, n_top_images):
    assert type(input_image) is np.ndarray and input_image.ndim == 3
    assert type(imbase) is skimage.io.ImageCollection
    assert type(sorted_indices) is np.ndarray and sorted_indices.ndim == 1
    assert type(n_top_images) is int and n_top_images <= 10

    plt.figure(1)
    plt.imshow(input_image)
    plt.axis('off')
    plt.show()
    fig = plt.figure(2)

    n_rows = 2
    n_cols = 5

    for image_index in xrange(n_top_images):
        sp = fig.add_subplot(n_rows, n_cols, image_index + 1)
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)
        sp.imshow(imbase[sorted_indices[image_index]])
        sp.autoscale(False)

    plt.show()
