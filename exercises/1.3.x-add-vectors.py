
import math
import numpy as np
from numba import cuda

# CUDA kernel.

@cuda.jit
def add(c, a, b):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if i < c.shape[0]:
        c[i] = a[i] + b[i]

# Test data.
n = 100000    
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros(n)

block_size = (16, )
grid_size = (math.ceil(n/block_size[0]), )

for i in range(100):
    add[grid_size, block_size](c, a, b)
    np.testing.assert_almost_equal(a+b, c)
