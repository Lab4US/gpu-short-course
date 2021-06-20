
import time
import math
import numpy as np
from numba import cuda, float32

@cuda.jit
def convolve_kernel(y, x, coeffs):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    
    if i >= y.shape[0]:
        return  
    value = float32(0.0)    
    n = min(coeffs.shape[0], i+1)
    for j in range(n):
        value += x[i-j]*coeffs[j]
    y[i] = value
        
        
def convolve(y, x, h):
    block_size = (256, )
    grid_size = (math.ceil(len(y)/block_size[0]), )
    convolve_kernel[grid_size, block_size](y, x, h)
    
# Tests

# Test 1
# x = np.array([0, 1, 2, 3, 4])
# h = np.array([0, 1, 2])
# y_gpu = cuda.device_array(len(x))

# convolve(y_gpu, x, h)
# np.testing.assert_equal(y_gpu.copy_to_host(), [0, 0, 1, 4, 7])

# # Test 1
# x = np.random.rand(1000)
# h = np.random.rand(30)
# y_gpu = cuda.device_array(len(x))
# convolve(y_gpu, x, h)
# np.testing.assert_equal(y_gpu.copy_to_host(), [0, 0, 1, 4, 7])

        
# Test data.
n = 1024*64*16*16 

for i in range(100):
    x_host = np.random.rand(n).astype(np.float32)
    h_host = np.random.rand(256).astype(np.float32)
    start = time.time()
    y_gpu = cuda.device_array(shape=(n,), dtype=np.float32) # np.zeros(n, dtype=np.float32)
    x_gpu = cuda.to_device(x_host)
    h_gpu = cuda.to_device(h_host)
    convolve(y_gpu, x_gpu, h_gpu)
    y_host = y_gpu.copy_to_host()
    end = time.time()
    print(f"Execution time: {end-start}")
