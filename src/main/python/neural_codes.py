import numpy as np
import caffe
import skimage
from skimage import io
from extra_functions import pairwise_distance, pca_apply, show_nearest


def caffe_image_transformer(shape, image_mean):
    assert type(shape) is tuple
    assert len(shape) == 4
    assert type(image_mean) is np.ndarray
    assert image_mean.ndim == 3
    assert shape[1] == image_mean.shape[0]

    image_mean = image_mean.transpose((1, 2, 0))
    im_height = shape[2]
    im_width = shape[3]
    image_mean = caffe.io.resize_image(image_mean, (im_height, im_width)).transpose((2, 0, 1))
    transformer = caffe.io.Transformer({'data': shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    # uncomment next line when use caffe.io.load_image()
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', image_mean)  # subtract the dataset-mean value in each channel

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

    for im in imbase:
        im = im.astype(np.float32)
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
                    max_patch_level_ref, max_patch_level_query, transformer, gpu_mode, u_reduce):
    assert type(input_image_full_name) is str
    assert type(imbase) is skimage.io.ImageCollection
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(neural_codes_query) is np.ndarray
    assert type(u_reduce) is np.ndarray and u_reduce.ndim == 2
    assert type(max_patch_level_ref) is int
    assert type(max_patch_level_query) is int
    assert isinstance(transformer, caffe.io.Transformer)  # old-style class
    assert type(gpu_mode) is bool
    layer_output_size = net.params[layer_name][1].data.shape[0]
    assert (u_reduce is None and neural_codes_query.shape[1] == layer_output_size) or \
           (type(u_reduce) is np.ndarray and u_reduce.ndim == 2) or \
           (type(u_reduce) is np.matrix)

    n_images = len(imbase.files)
    num_patches_per_image_ref = np.square(np.arange(1, max_patch_level_ref + 1)).sum()
    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query + 1)).sum()
    neural_codes_ref = np.empty([num_patches_per_image_ref, layer_output_size], dtype=np.float32)
    input_image_as_collection = skimage.io.ImageCollection(input_image_full_name)

    for patch_level_ref in xrange(1, max_patch_level_ref + 1):
        head = np.square(np.arange(patch_level_ref)).sum()
        tail = np.square(np.arange(1, patch_level_ref + 1)).sum()
        neural_codes_ref[head:tail] = nc_patch_codes(input_image_as_collection, net, layer_name, patch_level_ref,
                                                     transformer, gpu_mode)

    pairwise_distance_table = np.zeros((num_patches_per_image_ref, num_patches_per_image_query * n_images),
                                       dtype=np.float32)

    total_rows_num = neural_codes_query.shape[0]
    descr_vec_len = neural_codes_query.shape[1]

    if descr_vec_len < layer_output_size:
        neural_codes_ref = pca_apply(neural_codes_ref, u_reduce, descr_vec_len)

    # for more complicated algorithm: num_bytes_per_patch_nc = np.float32().nbytes * descr_vec_len
    rest = total_rows_num

    while rest > 0:
        used_num = min(rest, 10)  # here should be more complicated function, but we use a constant for the moment
        head = total_rows_num - rest
        tail = total_rows_num - rest + used_num
        rest -= used_num
        pairwise_distance_table[:, head:tail] = pairwise_distance(neural_codes_ref, neural_codes_query[head:tail])

    neural_codes_distances = nc_distances(pairwise_distance_table, (num_patches_per_image_ref,
                                                                    n_images,
                                                                    num_patches_per_image_query))
    sorted_indices = neural_codes_distances.argsort()
    n_nearest = min(n_images, 10)
    show_nearest(input_image_as_collection[0], imbase, sorted_indices, n_nearest)


def nc_distances(pairwise_distance_table, shape):
    assert type(pairwise_distance_table) is np.ndarray or type(pairwise_distance_table) is np.matrix
    assert pairwise_distance_table.ndim == 2
    assert type(shape) is tuple
    assert len(shape) == 3
    assert np.prod(pairwise_distance_table.shape) == np.prod(shape)

    pairwise_distance_table = pairwise_distance_table.reshape(shape)
    min_distance_table = np.min(pairwise_distance_table, axis=2)
    neural_codes_distances = np.mean(min_distance_table, axis=0)
    return neural_codes_distances


def read_imbase_nc(neural_codes_full_name_list, descr_vec_len, n_images, max_patch_level_query, gpu_mode):
    assert type(neural_codes_full_name_list) is list
    assert type(descr_vec_len) is int
    assert type(n_images) is int
    assert type(max_patch_level_query) is int
    assert type(gpu_mode) is bool

    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query + 1)).sum()
    neural_codes_list = [np.load(neural_codes_full_name) for neural_codes_full_name in neural_codes_full_name_list]
    neural_codes = np.zeros((n_images * num_patches_per_image_query, descr_vec_len), dtype=np.float32)

    for image_index in xrange(n_images):
        for patch_level in xrange(1, max_patch_level_query + 1):
            num_patches_per_im_level = patch_level ** 2
            head = image_index * num_patches_per_image_query + np.square(np.arange(patch_level)).sum()
            head_in_matrix = image_index * num_patches_per_im_level
            tail = image_index * num_patches_per_image_query + np.square(np.arange(1, patch_level + 1)).sum()
            tail_in_matrix = (image_index + 1) * num_patches_per_im_level
            neural_codes[head:tail] = neural_codes_list[patch_level - 1][head_in_matrix:tail_in_matrix]

    return neural_codes
