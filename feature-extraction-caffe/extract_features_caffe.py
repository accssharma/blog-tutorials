#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: Ashish Sharma
@institute: Boise State University
@description:
This file extracts features for input images specified as paths in a file from different layers
in CaffeNet/Alexnet/Googlenet/VGG19 models.
"""
# Imports

import sys
import pickle
import caffe
import numpy as np
import csv
   

# Method that returns batch of image of size "batch_size"
def get_this_batch(image_list, batch_index, batch_size):
    start_index = batch_index * batch_size
    next_batch_size = batch_size    
    image_list_size = len(image_list)
    if(start_index + batch_size > image_list_size):  
        reamaining_size_at_last_index = image_list_size - start_index
        next_batch_size = reamaining_size_at_last_index
    batch_index_indices = range(start_index, start_index+next_batch_size,1)
    return image_list[batch_index_indices]

def main(argv):
    gpu_id = 4
    extract_from_layer = "fc7"
    input_exp_file = "images.txt"
    model_def = "deploy.prototxt"
    pretrained_model = "blvc_alexnet.caffemodel"
    output_pkl_file_name = "out.pkl"
    batch_size = 10
    
    ext_file = open(input_exp_file, 'r')
    image_paths_list = [line.strip() for line in ext_file]    
    ext_file.close()
    
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id); 

    channel_swap_val = (2,1,0)
    raw_scale_value = 255
    
    images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list] 
    
    net = caffe.Net(model_def, pretrained_model, caffe.TEST)
    
    ## Set up transformer configuration
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1)) # HxWxC -> CxHxW
    transformer.set_channel_swap('data', channel_swap_val) # RGB -> BGR (2,1,0)
    transformer.set_raw_scale('data', raw_scale_value) 
    
    total_batch_nums = len(images_loaded_by_caffe)/batch_size    
    features_all_images = []
    images_loaded_by_caffe = np.array(images_loaded_by_caffe)
    for j in range(total_batch_nums+1):
	image_batch_to_process = get_this_batch(images_loaded_by_caffe, j, batch_size)
	num_images_being_processed = len(image_batch_to_process)
	data_blob_index = range(num_images_being_processed)
        net.blobs['data'].data[data_blob_index] = [transformer.preprocess('data', img) for img in image_batch_to_process]
        res = net.forward() # BEWARE: blobs arrays are overwritten
        features_for_this_batch = net.blobs[extract_from_layer].data[data_blob_index].copy()
        features_all_images.extend(features_for_this_batch)
    pkl_object = {"filename": image_paths_list, "data":features_all_images}
    output = open(output_pkl_file_name, 'wb')
    pickle.dump(pkl_object, output, 2)
    output.close()

if __name__ == "__main__":
	main(sys.argv[1:])
