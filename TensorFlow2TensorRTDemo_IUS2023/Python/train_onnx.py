# -*- coding: utf-8 -*-
"""

This example builds and trains a simple model on the MNIST dataset.
The model is then saved, converted into ONNX using tf2onnx for later TensorRT integration.
The input and output layer names are printed and will be needed for loading into TensorRT.
The onnx model is saved in the location of the executable:
    cpp/sample_mnist_data/simple_nn.onnx

@author: Hassan Nahas @ LITMUS
Modified from sample code by ONNX Project
"""

# SPDX-License-Identifier: Apache-2.0


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # modify for multi-gpu system, set to 0 for one gpu system

import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

########################################
# Creates the model.

model = tf.keras.models.Sequential([
  tf.keras.Input((28, 28,1),batch_size=1),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Softmax()
])

print(model.summary())
input_names = [n.name for n in model.inputs]
output_names = [n.name for n in model.outputs]
print('inputs:', input_names)
print('outputs:', output_names)

########################################
# Train the model on MNIST

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

########################################
# Testing the model.

input = x_test
expected = model.predict(input)
print(expected)

########################################
# Saves the model for ONNX conversion.

if not os.path.exists("simple_nn"):
    os.mkdir("simple_nn")
tf.keras.models.save_model(model, "simple_nn")

########################################
# Run the command line for ONNX conversion.

proc = subprocess.run('python -m tf2onnx.convert --saved-model simple_nn '
                      '--output ../cpp/sample_mnist_data/simple_nn.onnx --opset 15'.split(),
                      capture_output=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))
