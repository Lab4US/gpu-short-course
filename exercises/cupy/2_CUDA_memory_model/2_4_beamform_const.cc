// TODO declare __constant__ angles
// TODO declare __constant__ elements_x

extern "C"
__global__ void beamform(float *output, const float *rf,
                         const float *grid_z, const int nz,
                         const float *grid_x, const int nx,
                         const int nangles, const int nelements, const int nsamples,
                         const float speed_of_sound, const float sampling_frequency) {
    
    int z = blockIdx.x*blockDim.x + threadIdx.x; 
    int x = blockIdx.y*blockDim.y + threadIdx.y;
    
    // (x, z) - output pixel. Each thread computes a single output pixel.
    if (z >= nz || x >= nx) {
        return;
    }
    float pixel_z = grid_z[z]; // [m]
    float pixel_x = grid_x[x]; // [m]
    
    float tx_distance, rx_distance, angle, sample_number, propagation_time, pixel_value = 0.0f;
    int sample_number_int, tx_offset, rx_offset;
    
    for (int i = 0; i < nangles; ++i) {
        int tx_offset = i*nelements*nsamples;
        
        angle = angles[i];
        tx_distance = pixel_z*cosf(angle) + pixel_x*sinf(angle);
        
        for (int element = 0; element < nelements; ++element) {
            rx_distance = hypotf(pixel_x - elements_x[element], pixel_z);

            propagation_time = (tx_distance + rx_distance)/speed_of_sound;
            sample_number = propagation_time*sampling_frequency;
            
            if (sample_number >= 0.0f && sample_number <= (float)(nsamples - 1)) {
                rx_offset = element*nsamples;
                // Nearest-neighbour interpolation.
                sample_number_int = (int)roundf(sample_number);
                pixel_value += rf[tx_offset+rx_offset+sample_number_int];
            }
        }
        output[x*nz + z] = pixel_value;
    }
}