
import math
import numpy as np
from numba import cuda

# CUDA kernel.

@cuda.jit
def convolve_kernel(y, x, coeffs):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if i < y.shape[0]:
        value = 0.0    
        n = min(coeffs.shape[0], i+1)
        for j in range(n):
            value += x[i-j]*coeffs[j]
        y[i] = value
        
        
def convolve(y, x, h):
    block_size = (256, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    convolve_kernel[grid_size, block_size](y, x, h)
        
# Test data.
n = 100000
x_host = np.random.rand(n).astype(np.float32)
h_host = np.random.rand(64).astype(np.float32)
y_gpu = np.zeros(n, dtype=np.float32)

x_gpu = cuda.to_device(x_host)
h_gpu = cuda.to_device(h_host)

def convolve(y, x, h):
    block_size = (256, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    convolve_kernel[grid_size, block_size](y, x, h)

for i in range(100):
    convolve(y_gpu, x_gpu, h_gpu)
