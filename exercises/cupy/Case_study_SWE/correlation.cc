extern "C"
__global__ void correlation(
    float *lagImg,
    float *corImg,
    float *frames,
    const int maxPossibleLag,
    const int d,
    const int nFrames,    
    const int nx, 
    const int nz) 
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (z >= nz || x >= nx) {
        return;
    }
    
    int xd = (x+d) % nx;
    int maxlag = 0;
    float maxval = 0.0f;
    float value;
    int kFrame;
    int offset = nFrames/2;
    int n = 0;
    
    float m = 0;
    float md = 0;
    for (int iFrame = 0; iFrame < nFrames; ++iFrame){
        m += frames[z +  x*nz + iFrame*nx*nz];
        md += frames[z +  xd*nz + iFrame*nx*nz];
    }
    m = m/nFrames;
    md = md/nFrames;
    
    for (int lag = 0; lag < nFrames; ++lag){
        value = 0;
        for (int iFrame = 0; iFrame < nFrames; ++iFrame){
//             kFrame = (offset - lag + iFrame) % nFrames;
            kFrame = (offset + iFrame - lag);
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
//     if (maxlag >= maxPossibleLag){
//         maxlag = maxPossibleLag;
//     }

    lagImg[z + x*nz] = maxlag;
    corImg[z + x*nz] = maxval;
}