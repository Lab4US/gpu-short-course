extern "C"

__global__ void add_vectors_mod8_simple(float *c, float *a, float *b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Remember not to go outside the input data!
    if(i >= n) {
        return;
    }
    int r = i % 8;
    c[i] = r*a[i] + b[i];
}