#include <cupy/complex.cuh>

extern "C" __global__ 
void phase_shift(
    float *ps, 
    const complex<float> *iqFrames, 
    const int nBatch, 
    const int nBatchFrames,             
    const int nx, 
    const int nz, 
    const int step)                  
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z >= nz || x >= nx || t >= nBatch) {
        return;
    }
    
    /* Phase shift estimation */
        
    complex<float> sample;
    float ic, qc, ip, qp, pwr, nom = 0.0f, den = 0.0f;

    sample = iqFrames[z + x*nz + t*nx*nz*step];
    ic = real(sample);
    qc = imag(sample);
    
    for (int iFrame = 1; iFrame < nBatchFrames; iFrame++) {
        // previous I and Q values
        ip = ic;
        qp = qc;
        
        // current I and Q values
        sample = iqFrames[z + x*nz + t*nx*nz*step + iFrame*nz*nx];
        ic = real(sample);
        qc = imag(sample);
        
        den += ic*ip + qc*qp;
        nom += qc*ip - ic*qp;
    }
    ps[z + x*nz + t*nx*nz] = -atan2f(nom, den);
}