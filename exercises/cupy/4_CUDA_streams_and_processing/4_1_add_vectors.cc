extern "C"

__global__ void add_vectors(float *c, float *a, float *b, int n) {
    int i = // TODO

    // Remember not to go outside the input data!
    if(i >= n) {
        return;
    }
    
    c[i] = // TODO
}