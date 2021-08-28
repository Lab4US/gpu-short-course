extern "C"

__global__ void add_matrices(float *c, float *a, float *b, int height, int width) {
    int column = blockIdx.x*blockDim.x + threadIdx.x; // Solution
    int row    = blockIdx.y*blockDim.y + threadIdx.y; // Solution

    // Remember not to go outside the input data!
    if(column >= width || row >= height) {
        return;
    }
    // Flatten the input index.
    int i = row*width + column;

    c[i] = a[i] + b[i]; // Solution
}