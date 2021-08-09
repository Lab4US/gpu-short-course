extern "C"
__global__ void convolve1d_axis0(float *y, float *x, float *h, int height, int width, int n, int o) {
    int sample    = blockIdx.x*blockDim.x + threadIdx.x;
    int scanline  = blockIdx.y*blockDim.y + threadIdx.y;

    if(scanline >= width || sample >= height) {
        return;
    }
    
    float value = 0.0f;
    for(int j = 0; j < n; ++j) {
        int k = sample + o - j;
        if(k >= 0 && k < height) {
            // scanline, sample -> position in 1D array
            int input_sample = k*width + scanline;
            value += x[input_sample]*h[j];
        }
    }
    int output_sample = sample*width + scanline;
    y[output_sample] = value;
}