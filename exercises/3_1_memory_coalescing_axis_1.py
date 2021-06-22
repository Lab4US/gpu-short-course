

import math
import numpy as np
from numba import cuda, float32, int32
import cupy as cp


@cuda.jit
def convolve_gpu_kernel(y, x, h):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

    N = len(h)
    offset = int32(math.ceil(N/2)-1)
    
    HEIGHT = x.shape[0]
    WIDTH = x.shape[1]
    
    if i >= HEIGHT or j >= WIDTH:
        return
    
    value = float32(0.0)
    for k in range(N):
        l = i + offset - k
        if l >= 0 and l < HEIGHT:
            value += x[l, j]*h[k]
            
    y[i, j] = value
    
    
def convolve_gpu(y, x, h):
    block_size = (32, 32)
    height, width = x.shape
    # The left most index is the most quickly changing one.
    grid_size = (math.ceil(width/block_size[1]), math.ceil(height/block_size[0]))
    convolve_gpu_kernel[grid_size, block_size](y, x, h)
    
    
for i in range(10):
    x_host = np.random.rand(256, 256).astype(np.float32)
    h_host = np.random.rand(32).astype(np.float32)
    x_gpu = cuda.to_device(x_host)
    h_gpu = cuda.to_device(h_host)
    y_gpu = cuda.device_array(x_gpu.shape, dtype=x_gpu.dtype)
    convolve_gpu(y_gpu, x_gpu, h_gpu)
    y_host = y_gpu.copy_to_host()
