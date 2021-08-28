extern "C"
__global__ void decrement(float* data) {
    int i = threadIdx.x;
    data[i] = data[i] - 1;
}