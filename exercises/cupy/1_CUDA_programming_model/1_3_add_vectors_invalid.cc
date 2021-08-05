extern "C"

__global__ void add_vectors_invalid(float *c, float *a, float *b) {
    int i = threadIdx.x;
    c = 0; // ooops...
    c[i] = a[i] + b[i];
}