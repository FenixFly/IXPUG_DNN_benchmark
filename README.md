# IXPUG_DNN_benchmark
Scripts for benchmarking dnn frameworks for IXPUG report


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

### Start benchmark

```bash
 conda activate caffe
 cd <IXPUG_DNN_benchmark>/caffe_benchmark
 python3 caffe_benchmark.py -i ../datasets/imagenet/ -p ../models/squeezenet1.1.prototxt -m ../models/squeezenet1.1.caffemodel -ni 1000 -o true -of ./result/ -r result.csv
```


## OpenCV

Variant 1. Install opencv from conda

```bash
 conda create -n opencv
 conda activate opencv
 conda install -c conda-forge opencv
```

### Start benchmark

```bash
 conda activate opencv
 cd <IXPUG_DNN_benchmark>/opencv_benchmark
 python3 opencv_benchmark.py -i ../datasets/imagenet/ -p ../models/squeezenet1.1.prototxt -m ../models/squeezenet1.1.caffemodel -ni 1000 -o true -of ./result/ -r result.csv -w 227 -he 227 -s 1.0
```




