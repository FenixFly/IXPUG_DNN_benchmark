# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

TensorFlow benchmarking script

Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/tf_benchmark
mkdir results_classification
mkdir results_detection
python3 tf_benchmark.py -t classification -i ../datasets/imagenet/ -m ../models/resnet-50.pb -ni 1000 -o False -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -b 1
python3 tf_benchmark.py -t ssd300 -i ../datasets/pascal_voc/ -m ../models/ssd_300_vgg.ckpt -ni 1000 -o False -of ./results_detection/ -r ./results_detection/result.csv -w 300 -he 300 -b 1

Tensorflow realization of ResNet-50 was used from: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
TensorFlow realization of SSD300 was used from: https://github.com/balancap/SSD-Tensorflow

Created 04.10.2019

"""

import cv2
import tensorflow as tf
import os.path
import argparse
import numpy as np
from time import time

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an TensorFlow .pb model\
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
    return pictures / time

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

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph    
    
def load_network(model, width, height, task_type = 'classification'):
    
    if task_type == 'ssd300':
        slim = tf.contrib.slim
        import sys
        sys.path.append('ssd300')
        import ssd_vgg_300
        import ssd_vgg_preprocessing
        # Build ssd300 model
        isess = tf.InteractiveSession()

        net_shape=(height, width)
        input_tensor = tf.placeholder(tf.float32, shape=(None, height, width, 3))
        bbox_img = tf.constant([[0., 0., 1., 1.]])

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format='NHWC')):
            predictions, localisations, _, _ = ssd_net.net(input_tensor, is_training=False, reuse=reuse)

        # Restore SSD model.
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, model)
        
        return isess, (ssd_net, predictions, localisations, bbox_img), input_tensor
    else:
        graph = tf.Graph()
        sess = tf.InteractiveSession(graph = graph)
    
        # Import the TF graph
        with tf.gfile.GFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
                 
        # Define input tensor
        input_tensor = tf.placeholder(np.float32, shape = [None, width, height, 3], name='input') 
        tf.import_graph_def(graph_def, {'input': input_tensor})
            
    return sess, graph, input_tensor

def load_images(w, h, input_folder, numbers, mean = [0, 0, 0]):
    data = os.listdir(input_folder)
    counts = numbers
    if len(data)<numbers:
        counts = len(data)
    images = []
    for i in range(counts):
        image = cv2.imread(os.path.join(input_folder, data[i]))
        image = cv2.resize(image, (w, h))
        images.append(image)
        del image
    return images, counts

def tf_benchmark(sess, graph, input_tensor, width, height, number_iter, 
                 input_folder, need_output = False, output_folder = '', 
                 task_type = '', batch_size = 1, mean = [0, 0, 0]):
    filenames = os.listdir(input_folder)
    inference_times = []
    number_iter = (number_iter + batch_size - 1) // batch_size
    images, counts = load_images(width, height, input_folder, number_iter * batch_size, mean)
 
    if (task_type == 'ssd300'):
        images = np.array(images).astype(np.float32)
        ssd_net = graph[0]
        predictions = graph[1]
        localisations = graph[2]
        bbox_img = graph[3]
    else:
        output_tensor = graph.get_tensor_by_name("import/predict:0")
    
    #Warmup
    blob = np.array(images[0:batch_size])
    if (task_type == 'ssd300'):   
        rimg, rpredictions, rlocalisations, rbbox_img = sess.run([input_tensor, predictions, localisations, bbox_img], feed_dict = {input_tensor: blob})
    else:
        output = sess.run(output_tensor, feed_dict = {input_tensor: blob})
    
    t0_total = time()
    for i in range(number_iter):
        a = (i * batch_size) % len(images) 
        b = (((i + 1) * batch_size - 1) % len(images)) + 1
        
        blob = np.array(images[a:b])
            
        t0 = time()
        
        if (task_type == 'ssd300'):
            rimg, rpredictions, rlocalisations, rbbox_img = sess.run([input_tensor, predictions, localisations, bbox_img], feed_dict = {input_tensor: blob})
        else:
            output = sess.run(output_tensor, feed_dict = {input_tensor: blob})
        
        t1 = time()
    
        if (need_output == True and batch_size == 1):
            # Generate output name
            output_filename = str(os.path.splitext(os.path.basename(filenames[i]))[0])+'.npy'
            output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
            # Save output
            np.savetxt(output_filename, output)
    
        inference_times.append(t1 - t0)
    t1_total = time()
    inference_total_time = t1_total - t0_total
    return inference_times, inference_total_time

def main():	
    args = build_argparser().parse_args()
    create_result_file(args.result_file)

    # Load network	
    sess, graph, input_tensor = load_network(args.model, args.width, args.height, args.task_type)
    
    # Execute network
    inference_time, total_time = tf_benchmark(sess, graph, input_tensor, 
        args.width, args.height, args.number_iter, args.input_folder, args.output, args.output_folder, 
        args.task_type, args.batch_size, args.mean)

    # Write benchmark results
    inference_time = three_sigma_rule(inference_time)
    latency = calculate_latency(inference_time)
    fps = calculate_fps(args.number_iter, total_time)
    write_row(args.result_file, os.path.basename(args.model), args.batch_size,
              args.number_iter, latency, total_time, fps)

if __name__ == '__main__':
    main()