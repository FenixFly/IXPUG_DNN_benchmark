# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

OpenCV classification benchmarking script 


Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/opencv_benchmark

mkdir results_classification
python3 opencv_benchmark.py -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -s 1.0



Last modified 25.07.2019

"""

import cv2
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
    parser.add_argument('-w', '--width', help='Input tensor width', 
        required=True, type=int)
    parser.add_argument('-he', '--height', help='Input tensor height', 
        required=True, type=int)
    parser.add_argument('-s', '--scale', help='Input tensor values scaling', 
        required=True, type=float)
    parser.add_argument('-i', '--input_folder', help='Name of input folder',
        default='', type=str)
    parser.add_argument('-ni', '--number_iter', help='Number of inference \
        iterations', required=True, type=int)
    parser.add_argument('-o', '--output', help='Get output',
        default=False, type=bool)
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
        
    return parser

def load_network(model, config):
    net = cv2.dnn.readNet(model, config)
    
    return net
    
def opencv_benchmark(net, number_iter, input_folder, 
                    need_output = False, output_folder = ''):
    
    filenames = os.listdir(input_folder)
    filenames_size = len(filenames)
    inference_time = []
    for i in range(number_iter):
        image_name = os.path.join(input_folder, filenames[i % filenames_size])
        image = cv2.imread(image_name)
        blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (0, 0, 0))
        net.setInput(blob)
        
        
        t0 = time()
        preds = net.forward()
        t1 = time()
        
        if (need_output == True):
            # Generate output name
            output_filename = str(os.path.splitext(os.path.basename(image_name))[0])+'.npy'
            output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
            # Save output
            classification_output(preds, output_filename)
        inference_time.append(t1 - t0)
    return preds, inference_time

def classification_output(prob, output_file):
    prob = prob[0]
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
    net = load_network(args.model, args.proto)
    
    # Execute network
    pred, inference_time = opencv_benchmark(net, args.number_iter,
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
