import os
import caffe
import argparse
import numpy as np
import skimage
from skimage import io
from extra_functions import list_image_dir, gen_file_name, pca_matrix, pca_apply
from neural_codes import nc_compute, read_imbase_nc, nc_find_nearest, caffe_image_transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--imbase_name', type=str)
    parser.add_argument('--net_def', type=str)
    parser.add_argument('--net_file_name', type=str)
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--image_mean', type=str)
    parser.add_argument('--input_image_file_name', type=str)
    parser.add_argument('--max_patch_level_ref', type=int)
    parser.add_argument('--max_patch_level_query', type=int)
    parser.add_argument('--descr_vec_len', type=int)
    parser.add_argument('--gpu_mode', type=bool)
    args = parser.parse_args()

    root_dir = args.root_dir
    imbase_name = args.imbase_name
    net_def = args.net_def
    net_file_name = args.net_file_name
    layer_name = args.layer_name
    image_mean = args.image_mean
    input_image_file_name = args.input_image_file_name
    max_patch_level_ref = args.max_patch_level_ref
    max_patch_level_query = args.max_patch_level_query
    descr_vec_len = args.descr_vec_len
    gpu_mode = args.gpu_mode

    working_dir = os.path.join(root_dir, 'collections', imbase_name)
    imbase_path = os.path.join(working_dir, 'images')
    imbase_list = list_image_dir(imbase_path)
    imbase = skimage.io.imread_collection(imbase_list, conserve_memory=True)
    nc_path = os.path.join(working_dir, 'neuralcodes', 'numpy')
    net = caffe.Net(net_def, net_file_name, caffe.TEST)
    net_name = os.path.basename(net_file_name).split('.')[0]

    caffe.set_mode_gpu() if gpu_mode else caffe.set_mode_cpu()

    n_images = len(imbase_list)
    num_patches_per_image_query = np.square(np.arange(1, max_patch_level_query)).sum()
    image_mean = np.load(image_mean)
    transformer = caffe_image_transformer(net.blobs['data'].data.shape, image_mean)
    ext = '.npy'

    nc_full_name_list = [gen_file_name(nc_path, ext, imbase_name, net_name, 'nc', str(patch_level))
                         for patch_level in xrange(1, max_patch_level_query + 1)]

    if not os.path.isdir(nc_path):
        os.makedirs(nc_path)

    for patch_level in xrange(1, max_patch_level_query + 1):
        nc_full_name = nc_full_name_list[patch_level - 1]

        if not os.path.isfile(nc_full_name):
            print 'Features from {0} network on patch level #{1} are not available, extracting...'.\
                format(net_name, patch_level)

            neural_codes_on_level = nc_compute(imbase, net, 'fc6', patch_level, transformer, gpu_mode)
            np.save(nc_full_name, neural_codes_on_level)

            print 'Completed'

    result_path = os.path.join(working_dir, 'result')

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    input_image_dir = os.path.join(root_dir, 'references')
    input_image_full_name = os.path.join(input_image_dir, input_image_file_name)
    input_image_shortened_name = os.path.splitext(input_image_full_name)[0].split('/')[-1]
    result_file_name = '_'.join([imbase_name, input_image_shortened_name, 'nc']) + '.csv'
    result_full_name = os.path.join(result_path, result_file_name)
    layer_output_size = net.params[layer_name][1].data.shape[0]

    neural_codes = read_imbase_nc(nc_full_name_list, layer_output_size, n_images, max_patch_level_query, gpu_mode)

    if descr_vec_len < layer_output_size:

        nc_full_name_matrix_pca = gen_file_name(nc_path, ext, imbase_name, net_name, 'pca',
                                                str(max_patch_level_query))

        if not os.path.isfile(nc_full_name_matrix_pca):
            print 'PCA matrix for features from {0} network is not available, computing...'.format(net_name)

            u_reduce = pca_matrix(neural_codes)
            np.save(nc_full_name_matrix_pca, u_reduce)

            print 'Completed'
        else:
            u_reduce = np.load(nc_full_name_matrix_pca)

        nc_full_name_pca = gen_file_name(nc_path, ext, imbase_name, net_name, 'nc',
                                         str(max_patch_level_query), str(descr_vec_len))

        if not os.path.isfile(nc_full_name_pca):
            print 'PCA features from {0} network are not available, reducing dimensions...'.format(net_name)

            neural_codes = pca_apply(neural_codes, u_reduce, descr_vec_len)
            np.save(nc_full_name_pca, neural_codes)

            print 'Completed'

    print 'Searching for nearest images...'

    nc_find_nearest(input_image_full_name, imbase, net, layer_name, neural_codes,
                    max_patch_level_ref, max_patch_level_query, transformer, gpu_mode,
                    u_reduce if descr_vec_len < layer_output_size else None)

    print 'Done!'
