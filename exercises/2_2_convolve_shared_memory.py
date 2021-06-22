
import math
from numba import cuda, float32, int32
import cupy as cp
from tests import test_convolve, benchmark_convolve

THREAD_BLOCK_SIZE = 256

@cuda.jit
def convolve_gpu_kernel(y, x, h):
    i = cuda.grid(1)
    
    if i >= y.shape[0]:
        return
    
    M = len(x)
    N = len(h)
    
    x_shared = cuda.shared.array(shape=0, dtype=float32)
    SHARED_SIZE = cuda.blockDim.x+N-1
    OFFSET = int32(math.ceil(N/2)-1)
    
    # Copy a portion of data from global memory to shared memory.
    
    # The current position in the global memory.
    k = i-(N-1)+OFFSET 
    # The current position in the shared memory.
    k_shared = cuda.threadIdx.x 
    while k_shared < SHARED_SIZE:
        if k >= 0 and k < M:
            x_shared[k_shared] = x[k]
        else:
            x_shared[k_shared] = float32(0.0)
        k_shared += cuda.blockDim.x
        k        += cuda.blockDim.x

    cuda.syncthreads()
    
    k_shared = cuda.threadIdx.x+N-1
    value = float32(0.0)
    for j in range(N):
        value += x_shared[k_shared-j]*h[j]
        
    y[i] = value
    
def convolve_gpu(y, x, h):
    if y is None:
        y = cuda.device_array(x.shape, dtype=x.dtype)
    
    # Determine thread and block size.
    n_threads = min(THREAD_BLOCK_SIZE, len(y))
    block_size = (n_threads, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    
    # Determine shared memory size.
    N = len(h)
    SHARED_SIZE = THREAD_BLOCK_SIZE+N-1
    SHARED_SIZE_BYTES = SHARED_SIZE*y.dtype.itemsize
    
    # Execute the kernel.
    convolve_gpu_kernel[grid_size, block_size, cuda.default_stream(), SHARED_SIZE_BYTES](y, x, h)
    return y.copy_to_host()


benchmark_convolve(lambda x, h: convolve_gpu(None, x, h))
