#include <cupy/complex.cuh>

extern "C" __global__ 
void doppler(float *color, 
             float *power, 
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
    
    /* Color and Power estimation */
        
    complex<float> sample;
    float ic, qc, ip, qp, pwr, nom = 0.0f, den = 0.0f;

    sample = iqFrames[z + x*nz + t*nx*nz*step];
    ic = real(sample);
    qc = imag(sample);
    pwr = ic*ic + qc*qc;
    
    for (int iFrame = 1; iFrame < nBatchFrames; iFrame++) {
        // previous I and Q values
        ip = ic;
        qp = qc;
        
        // current I and Q values
        sample = iqFrames[z + x*nz + t*nx*nz*step + iFrame*nz*nx];
        ic = real(sample);
        qc = imag(sample);
        
        pwr += ic*ic + qc*qc;
        den += ic*ip + qc*qp;
        nom += qc*ip - ic*qp;
    }
    color[z + x*nz + t*nx*nz] = atan2f(nom, den);
    power[z + x*nz + t*nx*nz] = pwr/nBatchFrames;
}