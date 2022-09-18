#define IDX_4D(x, y, z, w, Y_SIZE, Z_SIZE, W_SIZE) \
    (x)*(Y_SIZE)*(Z_SIZE)*(W_SIZE) + (y)*(Z_SIZE)*(W_SIZE) + (z)*(W_SIZE) + (w)


#define IDX_3D(y, z, w, Z_SIZE, W_SIZE)             \
    (y)*(Z_SIZE)*(W_SIZE) + (z)*(W_SIZE) + (w)


__forceinline__
__device__ float interpLinear(const float *input, float sample, int nSamples) {
  float interpWgh = modff(sample, &sample);
  int intSample = int(sample);
  if(intSample >= nSamples-1) {
      // We skip the edge case where intSample == last sample, for the sake of simplicity.
      // Extrapolate with zeros.
      return 0.0f;
  }
  else {
      return input[intSample]*(1-interpWgh) + input[intSample+1]*interpWgh;
  }
}

/**
   Beamforms input data.
   
   @param output: array for the output data (nYPix, nXPix, nZPix)
   @param input: array with the input data (nTx, nRx, nSamples)
   @param txApodization: an array with TX apodization (binary) weights (nTx, nXPix, nZPix)
   @param rxApodization: an array with RX apodization weights (nRx, nYPix, nZPix)
   @param txDelays: an array of TX delays with shape (nTx, nXPix, nZPix), (s)
   @param rxDelays: an array of RX delays with shape (nRx, nYPix, nZPix), (s)
   @param initDelay: aperture center transmit delay
   @param nTx: number of transmissions (angles)
   @param nSamples: number of samples
   @param nRx: number of receive aperture elements
   @param nXPix: number of pixels along OX axis
   @param nYPix: number of pixels along OY axis
   @param nZPix: number of pixels along OZ axis
   @param fs: sampling frequency (Hz)
 */
extern "C"
__global__ void delayAndSumLUT(float *output, const float *input,
                            const unsigned char *txApodization, const float *rxApodization,
                            const float *txDelays, const float *rxDelays,
                            const float initDelay, 
                            const int nTx, const int nSamples, const int nRx,
                            const int nYPix, const int nXPix, const int nZPix,
                            const float fs) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;

    if (z >= nZPix || x >= nXPix || y >= nYPix) {
        return;
    }
    float pixelValue = 0.0f, value = 0.0f;
    float pixelWeight = 0.0f;
    float txDelay, rxDelay, txWeight, rxWeight, delay, delaySample;

    for(int tx = 0; tx < nTx; ++tx) {
        pixelValue = 0.0f;
        pixelWeight = 0.0f;
        txDelay = txDelays[IDX_3D(tx, x, z, nXPix, nZPix)];
        txWeight = txApodization[IDX_3D(tx, x, z, nXPix, nZPix)];
        if(txWeight == 1) {
            for(int rx = 0; rx < nRx; ++rx) {
                rxWeight = rxApodization[IDX_3D(rx, y, z, nYPix, nZPix)];
                if(rxWeight != 0.0f) {
                    rxDelay = rxDelays[IDX_3D(rx, y, z, nYPix, nZPix)];
                    delay = initDelay + txDelay + rxDelay;
                    delaySample = delay*fs;
                    value = interpLinear(&input[IDX_3D(tx, rx, 0, nRx, nSamples)], delaySample, nSamples);
                    pixelValue += value*rxWeight;
                    pixelWeight += rxWeight;
                }
            }
        }
        if(pixelWeight != 0.0f) {
            pixelValue = pixelValue/pixelWeight;
        }
        output[IDX_4D(tx, y, x, z, nYPix, nXPix, nZPix)] = pixelValue;
    }
}    
