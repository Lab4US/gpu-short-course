extern "C"
__global__ void increment(float* data) {
    int i = threadIdx.x;
    data[i] = data[i] + 1;
}