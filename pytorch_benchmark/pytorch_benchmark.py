# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

PyTorch benchmarking script

Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/pytorch_benchmark
mkdir results_classification
mkdir results_detection
python3 pytorch_benchmark.py -t classification -i ../datasets/imagenet/ -m ../models/resnet-50.pth -ni 1000 -o False -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -b 1
python3 pytorch_benchmark.py -t ssd300 -i ../datasets/pascal_voc/ -m ../models/ssd300.pth -ni 1000 -o False -of ./results_detection/ -r ./results_detection/result.csv -w 300 -he 300 -b 1

PyTorch realization of ResNet-50 was used from: https://pytorch.org/hub/pytorch_vision_resnet/
PyTorch realization of SSD300 was used from: https://github.com/amdegroot/ssd.pytorch

Created 07.10.2019

"""

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import cv2
import os.path
import argparse
import numpy as np
from time import time

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an PyTorch .pth model\
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input_folder', help='Name of input folder',
        default='', type=str)
    parser.add_argument('-ni', '--number_iter', help='Number of inference \
        iterations', required=True, type=int)
    parser.add_argument('-o', '--output', help='Get output',
        required=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
    parser.add_argument('-t', '--task_type', help='Task type: \
        classification / detection', default = 'classification', type=str)
    parser.add_argument('-me', '--mean', help='Input mean values', 
                        default = '[0 0 0]', type=str)
    parser.add_argument('-b', '--batch_size', help='batch size', 
        default=1, type=int)
    parser.add_argument('-w', '--width', help='Input tensor width', 
        required=True, type=int)
    parser.add_argument('-he', '--height', help='Input tensor height', 
        required=True, type=int)
    return parser
	
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
    if time > 0:
        return pictures / time
    else: 
        return -1

def create_result_file(filename):
    if os.path.isfile(filename):
        return
    file = open(filename, 'w')
    head = 'Model;Batch size;Device;IterationCount;Latency;Total time (s);FPS;'
    file.write(head + '\n')
    file.close()

def write_row(filename, net_name, batch_size, number_iter, latency, total_time, fps):
    row = '{};{};CPU;{};{:.3f};{:.3f};{:.3f}'.format(net_name, batch_size, number_iter, 
            latency, total_time, fps)
    file = open(filename, 'a')
    file.write(row + '\n')
    file.close()

def load_network(model, width, height, task_type = 'classification'):
    
    if task_type == 'ssd300':
        import sys
        sys.path.append('ssd300')
        from ssd import build_ssd
        net = build_ssd('test', 300, 21)    # initialize SSD
        net.load_weights(model)
    else:
        device = torch.device('cpu')
        net = torch.load(model, map_location=torch.device('cpu'))
        net.eval()
    
    return net

def load_images(w, h, input_folder, numbers):
    data = os.listdir(input_folder)
    counts = numbers
    if len(data)<numbers:
        counts = len(data)
    images = []
    
    preprocess = transforms.Compose([
            transforms.Resize((w,h)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    for i in range(counts):
        image = Image.open(os.path.join(input_folder, data[i])).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)
        images.append(image)
        del image
    images = torch.cat(images, 0)
    return images, counts

def pytorch_benchmark(net, width, height, number_iter, 
                 input_folder, need_output = False, output_folder = '', 
                 task_type = '', batch_size = 1):
    filenames = os.listdir(input_folder)
    inference_times = []
    number_iter = (number_iter + batch_size - 1) // batch_size
    images, counts = load_images(width, height, input_folder, number_iter * batch_size)
    
    t0_total = time()
    for i in range(number_iter):
        a = (i * batch_size) % len(images) 
        b = (((i + 1) * batch_size - 1) % len(images)) + 1
        
        blob = images[a:b]
        
        t0 = time()
        
        output = net(blob)
        
        t1 = time()
    
        if (need_output == True and batch_size == 1):
            # Generate output name
            output_filename = str(os.path.splitext(os.path.basename(filenames[i]))[0])+'.npy'
            output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
            # Save output
            print(output.shape)
            print(np.argmax(np.array(output)[0]))
            #np.savetxt(output_filename, output)
    
        inference_times.append(t1 - t0)
    t1_total = time()
    inference_total_time = t1_total - t0_total
    return inference_times, inference_total_time

def main():	
    args = build_argparser().parse_args()
    create_result_file(args.result_file)

    # Load network	
    net = load_network(args.model, args.width, args.height, args.task_type)
    
    # Execute network
    inference_time, total_time = pytorch_benchmark(net, 
        args.width, args.height, args.number_iter, args.input_folder, args.output, args.output_folder, 
        args.task_type, args.batch_size)

    # Write benchmark results
    inference_time = three_sigma_rule(inference_time)
    latency = calculate_latency(inference_time)
    fps = calculate_fps(args.number_iter, total_time)
    write_row(args.result_file, os.path.basename(args.model), args.batch_size,
              args.number_iter, latency, total_time, fps)

if __name__ == '__main__':
    main()