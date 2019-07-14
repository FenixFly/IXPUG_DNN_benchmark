# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

Caffe classification benchmarking script 


Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/caffe_benchmark
python3 caffe_benchmark.py -i ../datasets/imagenet/ -p ../models/squeezenet1.1.prototxt -m ../models/squeezenet1.1.caffemodel -ni 2 -o true -of ./result/ -r result.csv

Last modified 14.07.2019

"""

import cv2
import caffe
import os.path
import argparse
import numpy as np
from time import time

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-p', '--proto', help='Path to an .prototxt \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-m', '--model', help='Path to an .caffemodel file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input_folder', help='Name of input folder',
        default='', type=str)
    parser.add_argument('-ni', '--number_iter', help='Number of inference \
        iterations', required=True, type=int)
    parser.add_argument('-o', '--output', help='Get output',
        required=True, type=bool)
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
        
    return parser

def load_network(proto, model):
    caffe.set_mode_cpu()
    network = caffe.Net(proto, model, caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': network.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    return network, transformer

def load_image_to_network(image_path, net, transformer):
    
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

def prepare_image(image_path, input_dims, scale = 1.0, mean = [0.0, 0.0, 0.0]):
    
    image = cv2.imread(image_path, 1).astype(np.float32) - np.asarray(mean)
    image = image * scale
    image_size = image.shape
    image = cv2.resize(image, (input_dims[2], input_dims[3]))
    return image, image_size


def caffe_benchmark(net, transformer, number_iter, input_folder, 
                    need_output = False, output_folder = ''):
    
    filenames = os.listdir(input_folder)
    filenames_size = len(filenames)
    inference_time = []
    for i in range(number_iter):
        image_name = os.path.join(input_folder, filenames[i % filenames_size])
        load_image_to_network(image_name, net, transformer)
        
        t0 = time()
        out = net.forward()
        t1 = time()
        
        prob = out['prob']
        if (need_output):
            # Generate output name
            output_filename = str(os.path.splitext(os.path.basename(image_name))[0])+'.npy'
            output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
            # Save output
            classification_output(prob, output_filename)
        inference_time.append(t1 - t0)
    return prob, inference_time


def classification_output(prob, output_file):
    prob = prob[0,:,0,0]
    top = np.argmax(prob)
    np.savetxt(output_file, prob)


def three_sigma_rule(time):
    average_time = np.mean(time)
    sigm = np.std(time)
    upper_bound = average_time + (3 * sigm)
    lower_bound = average_time - (3 * sigm)
    valid_time = []
    for i in range(len(time)):
        if lower_bound <= time[i] <= upper_bound:
            valid_time.append(time[i])
    return valid_time

def calculate_average_time(time):
    average_time = np.mean(time)
    return average_time

def calculate_latency(time):
    time.sort()
    latency = np.median(time)
    return latency

def calculate_fps(pictures, time):
    return pictures / time

def create_result_file(filename):
    if os.path.isfile(filename):
        return
    file = open(filename, 'w')
    head = 'Model;Batch size;Device;IterationCount;Average time of single pass (s);Latency;FPS;'
    file.write(head + '\n')
    file.close()

def write_row(filename, net_name, number_iter, average_time, latency, fps):
    row = '{0};1;CPU;{1};{2:.3f};{3:.3f};{4:.3f}'.format(net_name, number_iter, 
           average_time, latency, fps)
    file = open(filename, 'a')
    file.write(row + '\n')
    file.close()


def main():
    args = build_argparser().parse_args()
    create_result_file(args.result_file)
    
    # Load network
    net, transformer= load_network(args.proto, args.model)
    net_input_shape = np.asarray(net.blobs['data'].shape, dtype = int)
    
    # Execute network
    pred, inference_time = caffe_benchmark(net, transformer, args.number_iter,
                                     args.input_folder, args.output,
                                     args.output_folder)

    # Write benchmark results
    inference_time = three_sigma_rule(inference_time)
    average_time = calculate_average_time(inference_time)
    latency = calculate_latency(inference_time)
    fps = calculate_fps(1, latency)
    write_row(args.result_file, os.path.basename(args.model), args.number_iter, 
              average_time, latency, fps)


if __name__ == '__main__':
    main()