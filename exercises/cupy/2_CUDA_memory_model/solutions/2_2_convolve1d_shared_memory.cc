 extern "C"
__global__ void convolve1d(float *y, float *x, float *h, int m, int n, int o, int shared_mem_size) {
    // 0. INITIALIZATION.
    // The begining is almost the same as previously.
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Declare shared memory usage.
    extern __shared__ float x_shared[];

    // 1. READ DATA FROM GLOBAL TO SHARED MEMORY.
    int k = i-(n-1)+o;                         // The current position in the global memory.
    int k_shared = threadIdx.x;                // The current position in the shared memory.
    
     // SOLUTION
    while(k_shared < shared_mem_size) {        // Until we hit the end of the shared memory area.
        if (k >= 0 && k < m) {
            x_shared[k_shared] = x[k];
        }     
        else {
            x_shared[k_shared] = 0.0f;
        }
        k_shared += blockDim.x;
        k        += blockDim.x;
    }

    // 2. WAIT FOR OTHER THREADS FROM THE BLOCK.
    __syncthreads();

    // 3. COMPUTE y[i]
    if(i >= m) {
        return;
    }   

    float value = 0.0f;

    k_shared = threadIdx.x + n - 1;
    for(int j = 0; j < n; ++j) {
        // SOLUTION
        value += x_shared[k_shared-j]*h[j];
    }
    y[i] = value;
}