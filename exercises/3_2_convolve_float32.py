
import math
import numpy as np
from numba import cuda, float32, int32
from tests import benchmark_convolve

@cuda.jit
def convolve_gpu_kernel(y, x, h):
    i = cuda.grid(1)
    M = len(x)
    N = len(h)
    offset = int32(math.ceil(N/2)-1)
    
    if i >= len(y):
        return
    
    value = float32(0.0)
    
    for j in range(N):
        k = i + offset - j
        if k >= 0 and k < M:
            value += x[k]*h[j]
    
    y[i] = value
    
def convolve_gpu(y, x, h):
    if y is None:
        y = cuda.device_array(x.shape, dtype=x.dtype)
    block_size = (256, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    convolve_gpu_kernel[grid_size, block_size](y, x, h)
    return y.copy_to_host()

benchmark_convolve(lambda x, h: convolve_gpu(None, x, h))
