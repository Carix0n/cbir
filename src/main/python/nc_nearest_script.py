import os
import caffe
import argparse
import numpy as np
import skimage
from skimage import io
from extra_functions import list_image_dir
from neural_codes import nc_compute, read_imbase_nc, nc_find_nearest, caffe_image_transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--imbase_name', type=str)
    parser.add_argument('--net_def', type=str)
    parser.add_argument('--net_file_name', type=str)
    parser.add_argument('--image_mean', type=str)
    parser.add_argument('--input_image_file_name', type=str)
    parser.add_argument('--max_patch_level_ref', type=int)
    parser.add_argument('--max_patch_level_query', type=int)
    parser.add_argument('--descr_vec_len', type=int)
    parser.add_argument('--gpu_mode', type=bool)
    args = parser.parse_args()

    working_dir = os.path.join(args.root_dir, 'collections', args.imbase_name)
    imbase_path = os.path.join(working_dir, 'images')
    imbase_list = list_image_dir(imbase_path)
    imbase = skimage.io.imread_collection(imbase_list, conserve_memory=True)
    nc_path = os.path.join(working_dir, 'neuralcodes')
    net = caffe.Net(args.net_def, args.net_file_name, caffe.TEST)
    net_name = os.path.basename(args.net_file_name).split('.')[0]

    caffe.set_mode_gpu() if args.gpu_mode else caffe.set_mode_cpu()

    n_images = len(imbase_list)
    num_patches_per_image_query = \
        np.square(np.arange(1, args.max_patch_level_query)).sum()
    transformer = caffe_image_transformer(net, args.image_mean)
    ext = '.npy'

    nc_full_name_list = [os.path.join(nc_path, '_'.join([args.imbase_name, net_name, 'nc', str(patch_level)]) + ext)
                         for patch_level in xrange(1, args.max_patch_level_query + 1)]

    if not os.path.isdir(nc_path):
        os.makedirs(nc_path)

    for patch_level in xrange(1, args.max_patch_level_query + 1):
        nc_full_name = nc_full_name_list[patch_level - 1]
        if not os.path.isfile(nc_full_name):
            print 'Features from {0} network on patch level #{1} are not available, extracting...'.\
                format(net_name, patch_level)
            neural_codes_on_level = nc_compute(imbase, net, 'fc6', patch_level, transformer, args.gpu_mode)
            np.save(nc_full_name, neural_codes_on_level)
            # alternative read/write through np.ndarray.tofile(), np.fromfile()
            print 'Completed'

    result_path = os.path.join(working_dir, 'result')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    input_image_dir = os.path.join(args.root_dir, 'references')
    input_image_full_name = os.path.join(input_image_dir, args.input_image_file_name)
    input_image_shortened_name = net_name = os.path.splitext(input_image_full_name)[0].split('/')[-1]
    result_file_name = '_'.join([args.imbase_name, input_image_shortened_name, 'nc']) + '.csv'
    result_full_name = os.path.join(result_path, result_file_name)
    neural_codes = read_imbase_nc(nc_full_name_list, args.descr_vec_len, n_images,
                                  args.max_patch_level_query, args.gpu_mode)

    print 'Searching for nearest images...'
    nc_find_nearest(input_image_full_name, imbase, net, 'fc6', neural_codes,
                    args.max_patch_level_ref, args.max_patch_level_query, transformer, args.gpu_mode)
    print 'Done!'
