# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

OpenVINO classification benchmarking script 


Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/openvino_benchmark
python3 openvino_benchmark_async.py -i ../datasets/imagenet/ -c ../models/resnet-50.xml -m ../models/resnet-50.bin -ni 1000 -o True -of ./result/ -r result.csv -s 1.0 -w 224 -he 224 -tn 1 -sn 1 -b 1
Last modified 21.07.2019

"""

import sys
import cv2
import os.path
import argparse
import logging as log
import numpy as np
import copy
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-m', '--model', help='Path to an .bin file \
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
        required=True, type=bool)
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
    parser.add_argument('-tn', '--thread_num', help='threads num', 
        required=True, type=int)
    parser.add_argument('-sn', '--stream_num', help='threads num', 
        required=True, type=int)
    parser.add_argument('-b', '--batch_size', help='batch size', 
        required=True, type=int)
    
    parser.add_argument('-d', '--device', help = 'Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default = 'CPU', type = str)
        
    return parser


def prepare_model(log, model, weights, cpu_extension, device_list, plugin_dir,
                  thread_num, stream_num):
    model_xml = model
    model_bin = weights
    if len(device_list) == 1:
        device = device_list[0]
    elif len(device_list) == 2:
        device = 'HETERO:{},{}'.format(device_list[0], device_list[1])
    else:
        log.error('Wrong count devices')
        sys.exit(1)
    log.info('Plugin initialization.');
    plugin = IEPlugin(device = device, plugin_dirs = plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)
    log.info('Loading network files:\n\t {0}\n\t {1}'.format(
        model_xml, model_bin))
    net = IENetwork(model = model_xml, weights = model_bin)
    if plugin.device == 'CPU':
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [ l for l in net.layers.keys() \
            if l not in supported_layers ]
        if len(not_supported_layers) != 0:
            log.error('Following layers are not supported by the plugin \
                for specified device {0}:\n {1}'.format(plugin.device,
                ', '.join(not_supported_layers)))
            log.error('Please try to specify cpu extensions library path in \
                sample\'s command line parameters using -l or --cpu_extension \
                command line argument')
            sys.exit(1)
    if thread_num is not None:
        if 'CPU' in device_list:
            plugin.set_config({'CPU_THREADS_NUM': str(thread_num)})
        else:
            log.error('Parameter : Number of threads is used only for CPU')
            sys.exit(1)
    if stream_num is not None:
        if 'CPU' in device_list:
            plugin.set_config({'CPU_THROUGHPUT_STREAMS': str(stream_num)})
        else:
            log.error('Parameter : Number of streams is used only for CPU')
            sys.exit(1)
    if len(device_list) == 2:
        plugin.set_config({'TARGET_FALLBACK': device})
        plugin.set_initial_affinity(net)
    return net, plugin

def load_images(model, input_folder):
    data = os.listdir(input_folder)
    n, c, h, w  = model.inputs[next(iter(model.inputs))].shape
    images = np.ndarray(shape = (len(data), c, h, w))
    for i in range(len(data)):
        image = cv2.imread(os.path.join(input_folder, data[i]))
        if (image.shape[:-1] != (h, w)):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        images[i] = image
    return images
    

def openvino_benchmark_async(exec_net, net, number, batch_size, input_folder, 
                    need_output = False, output_folder = ''):
    curr_request_id = 0
    prev_request_id  = 1
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    filenames = os.listdir(input_folder)
    filenames_size = len(filenames)
    images = load_images(net, input_folder)
    
    number_iter = (number + batch_size - 1) // batch_size
    t0 = time()
    first = True
    res = []
    
    for i in range(number_iter):
        
        a = (i * batch_size) % len(images) 
        b = (((i + 1) * batch_size - 1) % len(images)) + 1         
        
        im_batch = images[a : b:]
    
        if (a > b):
            im_batch = images[b : b+batch_size:]
        infer_request_handle = exec_net.start_async(request_id = curr_request_id, 
                                inputs = {input_blob : im_batch})
        if first == True:
            first = False
            prev_request_id, curr_request_id = curr_request_id, prev_request_id
            continue
        
        if exec_net.requests[prev_request_id].wait(-1) == 0:
            if need_output:
                res.append(copy.copy(exec_net.requests[prev_request_id].
                        outputs[next(iter(net.outputs))]))
        prev_request_id, curr_request_id = curr_request_id, prev_request_id
    if exec_net.requests[prev_request_id].wait(-1) == 0:
        if need_output:
            res.append(copy.copy(exec_net.requests[prev_request_id].
                    outputs[next(iter(net.outputs))]))
                
    t1 = time()
    inference_time = t1 - t0
    
    result = []
    for r_l1 in res:
        for r_l2 in r_l1:
            result.append(r_l2)
    res = np.asarray(result[0: len(images)])
    
    return res, inference_time

def classification_output(prob, output_file):
    np.savetxt(output_file, prob)

def create_result_file(filename):
    if os.path.isfile(filename):
        return
    file = open(filename, 'w')
    head = 'Model;Batch size;Device;IterationCount;Thread num;Stream num; Average time (s); FPS;'
    file.write(head + '\n')
    file.close()

def write_row(filename, net_name, number_iter, batch_size, thread_num, stream_num, average_time, fps):
    row = '{0};{1};CPU;{2};{3};{4};{5:.3f};{6:.3f}'.format(net_name, batch_size, 
           number_iter, thread_num, stream_num, average_time, fps)
    file = open(filename, 'a')
    file.write(row + '\n')
    file.close()

def calculate_fps_async(pictures, time):
    return pictures / time

def save_preds(preds, input_folder, output_folder):
    filenames = os.listdir(input_folder)
    filenames_size = len(filenames)
    
    for i in range(preds.shape[0]):
        image_name = os.path.join(input_folder, filenames[i % filenames_size])
        output_filename = str(os.path.splitext(os.path.basename(image_name))[0])+'.npy'
        output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
        # Save output
        classification_output(preds[i,:], output_filename)


def main():
    args = build_argparser().parse_args()
    log.basicConfig(format = '[ %(levelname)s ] %(message)s',
        level = log.INFO, stream = sys.stdout)
    create_result_file(args.result_file)
    
    # Load network
    net, plugin = prepare_model(log, args.config, args.model, '', 
                                ['CPU'], '', args.thread_num,
                                args.stream_num)
    net.batch_size = args.batch_size
    exec_net = plugin.load(network=net, num_requests = 2)
    
    # Execute network
    pred, inference_time = openvino_benchmark_async(exec_net, net, args.number_iter,
                                     args.batch_size, args.input_folder, args.output,
                                     args.output_folder)
    
    # Write prediction
    if args.output:
        save_preds(pred, args.input_folder, args.output_folder)
    
    
    # Write benchmark results
    fps = calculate_fps_async(args.number_iter, inference_time)
    write_row(args.result_file, os.path.basename(args.model), args.number_iter, 
              args.batch_size, args.thread_num, args.stream_num, inference_time, fps)
    
    del exec_net
    del net

if __name__ == '__main__':
    main()