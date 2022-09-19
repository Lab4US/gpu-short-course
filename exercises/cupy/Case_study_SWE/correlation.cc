extern "C"
__global__ void correlation(
    float *lagImg,
    float *corImg,
    float *frames,
    const int d,
    const int nFrames,    
    const int nx, 
    const int nz) 
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (z >= nz || x >= nx-d) {
        return;
    }
    
    int xd = (x+d);
    int maxlag = 0;
    float maxval = 0;
    float value;
    int kFrame;
    int offset = nFrames/2;
    int n;
    
    // estimate mean value of processed signals for DC removal
    float m = 0;
    float md = 0;
    for (int iFrame = 0; iFrame < nFrames; ++iFrame){
         m += frames[z +  x*nz + iFrame*nx*nz];
        md += frames[z + xd*nz + iFrame*nx*nz];
    }
    m = m/nFrames;
    md = md/nFrames;
    
    // estimate max correlation and corresponding lag
    for (int lag = 0; lag < nFrames; ++lag){
        value = 0;
        n = 0;
        for (int iFrame = 0; iFrame < nFrames; ++iFrame){
            kFrame = (offset - lag + iFrame);
            if(kFrame >= 0 && kFrame < nFrames) {
                value += (frames[z +  x*nz + iFrame*nx*nz] - m)
                       * (frames[z + xd*nz + kFrame*nx*nz] - md);
                n += 1;
            }
        }
        value = value/n;
        if (value > maxval){
            maxval = value;
            maxlag = abs(lag - offset);
        }        
    }

    lagImg[z + x*nz] = maxlag;
    corImg[z + x*nz] = maxval;
}