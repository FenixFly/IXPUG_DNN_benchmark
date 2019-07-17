# IXPUG_DNN_benchmark
Scripts for benchmarking dnn frameworks for IXPUG report

Clone this repository to destination machine.

## Get networks

```bash
 <openvino_dir>/bin/setupvars.sh
 python3 <openvino_dir>/deployment_tools/tools/model_downloader/downloader.py -- name resnet-50
 python3 <openvino_dir>/deployment_tools/tools/model_downloader/downloader.py -- name ssd300
 
 python3 <openvino_dir>/deployment_tools/model_optimizer/mo.py --input_model <resnet50_folder>/resnet-50.caffemodel --input_proto <resnet50_folder>/resnet-50.prototxt 
 
 python3 <openvino_dir>/deployment_tools/model_optimizer/mo.py --input_model <ssd300_folder>/ssd300.caffemodel --input_proto <ssd300_folder>/ssd300.prototxt 
 
```

## Prepare software


The easiest way to avoid dependency hell is using seperate environments created by Conda. 

Download Miniconda here [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and install.

Create new environment for every framework, for example
```bash
 conda create -n caffe [python=3.6]
```

Next, activate environment

```bash
 conda activate caffe
```

Next, install framework and run benchmark.


## Caffe

Variant 1. Install caffe from conda

```bash
 conda create -n caffe
 conda activate caffe
 conda install -c intel caffe
```

### Start benchmark classifiction

```bash
 conda activate caffe
 cd <IXPUG_DNN_benchmark>/caffe_benchmark
 mkdir result
 python3 caffe_benchmark.py -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -o true -of ./result/ -r result.csv
```

## OpenCV

Variant 1. Install opencv from conda

```bash
 conda create -n opencv
 conda activate opencv
 conda install -c conda-forge opencv
```

### Start benchmark classifiction

```bash
 conda activate opencv
 cd <IXPUG_DNN_benchmark>/opencv_benchmark
 mkdir result
 python3 opencv_benchmark.py -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -o true -of ./result/ -r result.csv -w 224 -he 224
```




