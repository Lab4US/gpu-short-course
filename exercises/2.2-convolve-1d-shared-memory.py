
import math
import numpy as np
from numba import cuda, float32

thread_block_size_deafult = 256


@cuda.jit
def convolve_kernel(y, x, h):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    
    if i >= y.shape[0]:
        return
    
    filter_size = len(h)
    x_sm = cuda.shared.array(shape=0, dtype=float32)
    sm_size = cuda.blockDim.x+filter_size-1
    
    # Copy a portion global memory data to shared memory.
    k = i - (filter_size-1) # The current position in the global memory.
    k_sm = cuda.threadIdx.x # The current position in the shared memory.  
    while k_sm < sm_size:
        if k < 0:
            x_sm[k_sm] = float32(0.0)
        else:
            x_sm[k_sm] = x[k]
        k_sm += cuda.blockDim.x
        k    += cuda.blockDim.x

    cuda.syncthreads()
    
    k_sm = cuda.threadIdx.x+filter_size-1
    value = float32(0.0)
    for j in range(filter_size):
        value += x_sm[k_sm-j]*h[j]
    y[i] = value

    
def convolve(y, x, h):
    thread_block_size = min(thread_block_size_deafult, len(y))
    block_size = (thread_block_size, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    filter_size = len(h)
    shared_memory_size = thread_block_size+filter_size-1
    shared_memory_size_bytes = shared_memory_size*y.dtype.itemsize
    convolve_kernel[grid_size, block_size, cuda.default_stream(), shared_memory_size_bytes](y, x, h)  
    
# Test data.
n = 1024*64*16*16

for i in range(20):
    x_host = np.random.rand(n).astype(np.float32)
    h_host = np.random.rand(256).astype(np.float32)
    y_gpu = cuda.device_array(shape=(n,), dtype=np.float32)
    x_gpu = cuda.to_device(x_host)
    h_gpu = cuda.to_device(h_host)
    convolve(y_gpu, x_gpu, h_gpu)
    y_host = y_gpu.copy_to_host()
