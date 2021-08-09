extern "C"
__global__ void convolve1d(float *y, float *x, float *h, int m, int n, int o) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= m) {
        return;
    }

    float value = 0.0f;
    for(int j = 0; j < n; ++j) {
        // TODO
    }
    y[i] = value;
}