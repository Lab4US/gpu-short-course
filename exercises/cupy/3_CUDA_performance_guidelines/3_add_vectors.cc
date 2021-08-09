extern "C"

__global__ void add_vectors(float *c, float *a, float *b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Remember not to go outside the input data!
    if(i >= n) {
        return;
    }
    
    c[i] = a[i] + b[i];
}