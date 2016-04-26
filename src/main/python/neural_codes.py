import os
import numpy as np
import caffe
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
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer


def nc_patch_codes(imbase_files, net, layer_name, patch_level, image_mean_path, gpu_mode):
    assert type(imbase_files) is list
    assert type(net) is caffe.Net
    assert type(patch_level) is int
    assert type(layer_name) is str
    assert type(gpu_mode) is bool

    num_patches_per_image = patch_level ** 2
    n_images = len(imbase_files)
    total_patches_num = n_images * num_patches_per_image
    neural_codes = np.empty((total_patches_num, net.params[layer_name][1].data.shape[0]), dtype=np.float32)
    n_channels = net.blobs['data'].data.shape[1]
    net_input_height = net.blobs['data'].data.shape[2]
    net_input_width = net.blobs['data'].data.shape[3]
    patch_array = np.empty((n_channels, net_input_height, net_input_width, total_patches_num), dtype=np.float32)  # first 3 coordinates, e.g. (3, 227, 227)
    transformer = caffe_image_transformer(net, image_mean_path)
    cur_index = 0

    for image_index in xrange(n_images):
        file_name = imbase_files[image_index]
        im = caffe.io.load_image(file_name)  # also skimage contains imread_collection function for block reading
        im_height = im.shape[0]
        im_width = im.shape[1]
        for patch_ver_index in xrange(1, patch_level + 1):
            ver_lower_bound = int(round(float(patch_ver_index - 1) / float(patch_level + 1) * float(im_height)))
            ver_upper_bound = int(round(float(patch_ver_index + 1) / float(patch_level + 1) * float(im_height)))
            for patch_hor_index in xrange(1, patch_level + 1):
                hor_lower_bound = int(round(float(patch_hor_index - 1) / float(patch_level + 1) * float(im_width)))
                hor_upper_bound = int(round(float(patch_hor_index + 1) / float(patch_level + 1) * float(im_width)))
                patch = im[ver_lower_bound:ver_upper_bound, hor_lower_bound:hor_upper_bound, :]
                patch_array[:, :, :, cur_index] = transformer.preprocess('data', patch)
                cur_index += 1

    for patch_index in xrange(total_patches_num):
        net.blobs['data'].data[...] = patch_array[:, :, :, patch_index]
        net.forward()
        cnn_output = net.blobs[layer_name].data[0]
        neural_codes[patch_index] = cnn_output  # initially there was a .copy(), but getting rid of it is still works fine

    return neural_codes


def nc_compute(imbase_path, imbase_files, net, layer_name, patch_level, image_mean_path, gpu_mode):
    assert type(imbase_path) is str
    assert type(imbase_files) is list
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(patch_level) is int
    assert type(gpu_mode) is bool

    n_images = len(imbase_files)
    imbase_files = [os.path.join(imbase_path, image_file) for image_file in imbase_files]

    num_patches_per_image = patch_level ** 2
    neural_codes = np.empty((n_images * num_patches_per_image, net.params[layer_name][1].data.shape[0]), dtype=np.float32)
    rest = n_images

    while rest > 0:
        used_num = min(rest, 10)  # here should be more complicated function, but we use a constant for the moment
        head = n_images - rest
        tail = n_images - rest + used_num
        rest -= used_num
        neural_codes[(head * num_patches_per_image):(tail * num_patches_per_image)] = \
            nc_patch_codes(imbase_files[head:tail], net, layer_name, patch_level, image_mean_path, gpu_mode)

    return neural_codes


def nc_find_nearest(input_image_file_name, imbase_path, imbase_files, net, layer_name, neural_codes_query, max_patch_level_ref, max_patch_level_query, image_mean_path, gpu_mode):
    assert type(input_image_file_name) is str
    assert type(imbase_path) is str
    assert type(imbase_files) is list
    assert type(net) is caffe.Net
    assert type(layer_name) is str
    assert type(neural_codes_query) is np.ndarray
    assert type(max_patch_level_ref) is int
    assert type(max_patch_level_query) is int
    assert type(gpu_mode) is bool

    n_images = len(imbase_files)
    num_patches_per_image_ref = np.square(np.arange(1, max_patch_level_ref + 1)).sum()
    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query + 1)).sum()
    neural_codes_ref = np.empty([num_patches_per_image_ref, net.params[layer_name][1].data.shape[0]], dtype=np.float32)

    for patch_level_ref in xrange(1, max_patch_level_ref + 1):
        head = np.square(np.arange(patch_level_ref)).sum()
        tail = np.square(np.arange(1, patch_level_ref + 1)).sum()
        neural_codes_ref[head:tail] = nc_patch_codes([input_image_file_name], net, layer_name, patch_level_ref, image_mean_path, gpu_mode)

    pairwise_dist_table = np.zeros((num_patches_per_image_ref, n_images * num_patches_per_image_query), dtype=np.float32)
    # here we should use PCA
    total_rows_num = neural_codes_query.shape[0]
    # for more complicated algorithm: num_bytes_per_patch_nc = np.float32().nbytes * neural_codes_ref.shape[1]
    rest = total_rows_num

    while rest > 0:
        used_num = min(rest, 10)  # here should be more complicated function, but we use a constant for the moment
        head = total_rows_num - rest
        tail = total_rows_num - rest + used_num
        rest -= used_num
        pairwise_dist_table[:, head:tail] = pairwise_distance(neural_codes_ref, neural_codes_query[head:tail])

    neural_codes_distances = np.min(pairwise_dist_table.reshape([num_patches_per_image_ref, num_patches_per_image_query, n_images]), axis=1).mean(axis=0)
    sorted_indices = neural_codes_distances.argsort()
    n_nearest = min(n_images, 10)
    show_nearest(input_image_file_name, imbase_path, imbase_files, sorted_indices, n_nearest)


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
        head_sum = 0
        tail_sum = 0
        for patch_level_query in xrange(1, max_patch_level_query + 1):
            prev_level_size = (patch_level_query - 1) ** 2
            cur_level_size = patch_level_query ** 2
            head_sum += prev_level_size
            tail_sum += cur_level_size
            head = image_index * num_patches_per_image_query + head_sum
            tail = image_index * num_patches_per_image_query + tail_sum
            neural_codes[head:tail] = neural_codes_list[patch_level_query - 1][(image_index * cur_level_size):((image_index + 1) * cur_level_size)]

    return neural_codes
