extern "C"

__global__ void add_matrices(float *c, float *a, float *b, int height, int width) {
    int column = // TODO
    int row    = // TODO

    // Remember not to go outside the input data!
    if(column >= width || row >= height) {
        return;
    }
    // Flatten the input index.
    int i = row*width + column;

    c[i] = // TODO
}