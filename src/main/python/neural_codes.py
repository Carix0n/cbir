import os
import numpy as np
import caffe
import skimage
from skimage import io
from extra_functions import pairwise_distance, show_nearest


def caffe_image_transformer(net, image_mean_path):
    assert type(net) is caffe.Net

    mean = np.load(image_mean_path).transpose((1, 2, 0))
    im_height = net.blobs['data'].data.shape[2]
    im_width = net.blobs['data'].data.shape[3]
    mean = caffe.io.resize_image(mean, (im_height, im_width)).transpose((2, 0, 1))
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean)  # subtract the dataset-mean value in each channel
    # uncomment next line with usage of caffe.io.load_image()
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer


def nc_patch_codes(imbase, net, layer_name, patch_level, transformer, gpu_mode):
    assert type(imbase) is skimage.io.ImageCollection
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(patch_level) is int
    assert isinstance(transformer, caffe.io.Transformer)  # old-style class
    assert type(gpu_mode) is bool

    n_images = len(imbase.files)
    num_patches_per_image = patch_level ** 2
    total_patches_num = n_images * num_patches_per_image
    neural_codes = np.empty((total_patches_num, net.params[layer_name][1].data.shape[0]), dtype=np.float32)
    patch_index = 0

    for image_index in xrange(n_images):
        im = imbase[image_index]
        im_height = im.shape[0]
        im_width = im.shape[1]
        for patch_ver_index in xrange(1, patch_level + 1):
            ver_lower_bound = int(round(float(patch_ver_index - 1) / float(patch_level + 1) * float(im_height)))
            ver_upper_bound = int(round(float(patch_ver_index + 1) / float(patch_level + 1) * float(im_height)))
            for patch_hor_index in xrange(1, patch_level + 1):
                hor_lower_bound = int(round(float(patch_hor_index - 1) / float(patch_level + 1) * float(im_width)))
                hor_upper_bound = int(round(float(patch_hor_index + 1) / float(patch_level + 1) * float(im_width)))
                patch = im[ver_lower_bound:ver_upper_bound, hor_lower_bound:hor_upper_bound, :]
                net.blobs['data'].data[...] = transformer.preprocess('data', patch)
                net.forward()
                neural_codes[patch_index] = net.blobs[layer_name].data[0]
                patch_index += 1

    return neural_codes


def nc_compute(imbase, net, layer_name, patch_level, transformer, gpu_mode):
    assert type(imbase) is skimage.io.ImageCollection
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(patch_level) is int
    assert isinstance(transformer, caffe.io.Transformer)  # old-style class
    assert type(gpu_mode) is bool

    n_images = len(imbase.files)
    num_patches_per_image = patch_level ** 2
    neural_codes = np.empty((n_images * num_patches_per_image, net.params[layer_name][1].data.shape[0]),
                            dtype=np.float32)
    rest = n_images

    while rest > 0:
        used_num = min(rest, 10)  # here should be more complicated function, but we use a constant for the moment
        head = n_images - rest
        tail = n_images - rest + used_num
        rest -= used_num
        neural_codes[(head * num_patches_per_image):(tail * num_patches_per_image)] = \
            nc_patch_codes(imbase[head:tail], net, layer_name, patch_level, transformer, gpu_mode)

    return neural_codes


def nc_find_nearest(input_image_full_name, imbase, net, layer_name, neural_codes_query,
                    max_patch_level_ref, max_patch_level_query, transformer, gpu_mode):
    assert type(input_image_full_name) is str
    assert type(imbase) is skimage.io.ImageCollection
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(neural_codes_query) is np.ndarray
    assert type(max_patch_level_ref) is int
    assert type(max_patch_level_query) is int
    assert isinstance(transformer, caffe.io.Transformer)  # old-style class
    assert type(gpu_mode) is bool

    n_images = len(imbase.files)
    num_patches_per_image_ref = np.square(np.arange(1, max_patch_level_ref + 1)).sum()
    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query + 1)).sum()
    neural_codes_ref = np.empty([num_patches_per_image_ref, net.params[layer_name][1].data.shape[0]], dtype=np.float32)
    input_image_as_collection = skimage.io.ImageCollection(input_image_full_name)

    for patch_level_ref in xrange(1, max_patch_level_ref + 1):
        head = np.square(np.arange(patch_level_ref)).sum()
        tail = np.square(np.arange(1, patch_level_ref + 1)).sum()
        neural_codes_ref[head:tail] = nc_patch_codes(input_image_as_collection, net, layer_name, patch_level_ref,
                                                     transformer, gpu_mode)

    pairwise_distance_table = np.zeros((num_patches_per_image_ref, num_patches_per_image_query * n_images),
                                       dtype=np.float32)
    # here we should use PCA
    total_rows_num = neural_codes_query.shape[0]
    # for more complicated algorithm: num_bytes_per_patch_nc = np.float32().nbytes * neural_codes_ref.shape[1]
    rest = total_rows_num

    while rest > 0:
        used_num = min(rest, 10)  # here should be more complicated function, but we use a constant for the moment
        head = total_rows_num - rest
        tail = total_rows_num - rest + used_num
        rest -= used_num
        pairwise_distance_table[:, head:tail] = pairwise_distance(neural_codes_ref, neural_codes_query[head:tail])

    pairwise_distance_table = pairwise_distance_table.reshape((num_patches_per_image_ref,
                                                               num_patches_per_image_query,
                                                               n_images))
    min_distance_table = np.min(pairwise_distance_table, axis=1)
    neural_codes_distances = np.mean(min_distance_table, axis=0)
    sorted_indices = neural_codes_distances.argsort()
    n_nearest = min(n_images, 10)
    show_nearest(input_image_as_collection[0], imbase, sorted_indices, n_nearest)


def read_imbase_nc(neural_codes_full_name_list, descr_vec_len, n_images, max_patch_level_query, gpu_mode):
    assert type(neural_codes_full_name_list) is list
    assert type(descr_vec_len) is int
    assert type(n_images) is int
    assert type(max_patch_level_query) is int
    assert type(gpu_mode) is bool

    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query + 1)).sum()
    neural_codes_list = [np.load(neural_codes_full_name) for neural_codes_full_name in neural_codes_full_name_list]
    neural_codes = np.empty((n_images * num_patches_per_image_query, descr_vec_len), dtype=np.float32)

    for image_index in xrange(n_images):
        for patch_level in xrange(1, max_patch_level_query + 1):
            num_patches_per_im_level = patch_level ** 2
            head = image_index * num_patches_per_image_query + np.square(np.arange(patch_level)).sum()
            head_in_matrix = image_index * num_patches_per_im_level
            tail = image_index * num_patches_per_image_query + np.square(np.arange(1, patch_level + 1)).sum()
            tail_in_matrix = (image_index + 1) * num_patches_per_im_level
            neural_codes[head:tail] = neural_codes_list[patch_level - 1][head_in_matrix:tail_in_matrix]

    return neural_codes
