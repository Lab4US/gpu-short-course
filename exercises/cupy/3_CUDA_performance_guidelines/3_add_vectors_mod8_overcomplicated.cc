extern "C"

__global__ void add_vectors_mod8_overcomplicated(float *c, float *a, float *b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Remember not to go outside the input data!
    if(i >= n) {
        return;
    }
    
    if(i % 8 == 0) {
        c[i] = b[i];
    }
    else if(i % 8 == 1) {
        c[i] = 1*a[i] + b[i];
    }
    else if(i % 8 == 2) {
        c[i] = 2*a[i] + b[i];
    }
    else if(i % 8 == 3) {
        c[i] = 3*a[i] + b[i];
    }
    else if(i % 8 == 4) {
        c[i] = 4*a[i] + b[i];
    }
    else if(i % 8 == 5) {
        c[i] = 5*a[i] + b[i];       
    }
    else if(i % 8 == 6) {
        c[i] = 6*a[i] + b[i];
    }
    else if(i % 8 == 7) {
        c[i] = 7*a[i] + b[i];

    }
}