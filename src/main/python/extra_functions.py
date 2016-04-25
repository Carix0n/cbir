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
    files = [elem for elem in files if os.path.isfile(os.path.join(imbase_path, elem))]
    return files


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


def show_nearest(input_image_file, imbase_path, imbase_files, sorted_indices, n_top_images):
    assert type(input_image_file) is str
    assert type(imbase_path) is str
    assert type(imbase_files) is list
    assert type(sorted_indices) is np.ndarray and sorted_indices.ndim == 1
    assert type(n_top_images) is int and n_top_images <= 10

    input_image = skimage.io.imread(input_image_file)
    plt.figure(1)
    plt.imshow(input_image)
    plt.axis('off')
    plt.show()
    fig = plt.figure(2)
    n_rows = 2
    n_cols = 5
    for image_index in xrange(n_top_images):
        sp = fig.add_subplot(n_rows, n_cols, image_index + 1)
        image = skimage.io.imread(os.path.join(imbase_path, imbase_files[sorted_indices[image_index]]))
        sp.axes.get_xaxis().set_visible(False)
        sp.axes.get_yaxis().set_visible(False)
        sp.imshow(image)
        sp.autoscale(False)

    plt.show()
