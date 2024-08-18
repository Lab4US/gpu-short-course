Sample project for TensorFlow-TensorRT integration via ONNX.

The project:
-trains a neural network model on the MNIST dataset in TensorFlow,
-exports it as an ONNX model using tf2onnx
-loads the ONNX model in C++ TensorRT
-demonstrates and times GPU-based inference in TensorRT on the MNIST test case.

The directory contains two modules:
Python: Trains a neural network on the MNIST dataset using TensorFlow-Python, exports it as an onnx model file.
 - train_onnx.py: python script that does the training and onnx conversion. 

Cpp: Contains a VS project for GPU inference using TensorRT with the ONNX model saved by the python module.
 - sample_mnist_data: directory for onnx neural network file and sample MNIST data for inference testing using TensorRT
 - TensorRT_ONNXMNIST: VS project for GPU inference using TensorRT
	- TensorRT_ONNX_MNIST.sln: VS solution file for project.
	- main.cpp: main c++ file for loading the ONNX model and GPU inference using TensorRT API.
	- TensorRT_ONNX_MNIST.vcxproj: project file for c++ project. Must have paths to TensorRT, Cudnn, CUDA defined
	- logger.cpp: helper code for logging from TensorRT API
	- output_example.txt: shows sample output from a successful run.
	- x64/Release: directory for compile executable and required dlls. Executable needs CUDNN and TensorRT Dlls to work. They can be copied here.
		- TensorRT_ONNX_MNIST: target executable

Requirements:
-Python
-Visual Studio (C++)
-CUDA
-CUDNN
-TensorRT
-TensorFlow
-Numpy
-tf2onnx

Tested Environment:
-Python 3.9.16
-Visual Studio (C++) Windows SDK 10.0.19041.0, Visual Studio 2017 v141 Toolset
-CUDA v11.7
-CUDNN 8.9.4
-TensorRT 8.6.1.6
-TensorFlow 2.10.1
-Numpy 1.23.5
-tf2onnx 1.14.0

Created by Hassan Nahas @ LITMUS based on sample code from NVIDIA-TensorRT and ONNX Project

See module files for license details.
