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
   @param txDelays: an array of TX delays with shape (nTx, nXPix, nZPix)
   @param rxDelays: an array of RX delays with shape (nRx, nYPix, nZPix)
 */
extern "C"
__global__ void delayAndSumOTF(float *output, const float *input,
                            const unsigned char *txApodization, const float *rxApodization,
                            // Parameters needed to determine TX and RX delays
                            const float speedOfSound, 
                            const float pitch, 
                            const int nElements,  
                            const float* txAngles, 
                            const float xStart, const float dx,
                            const float yStart, const float dy,
                            const float zStart, const float dz,   
                            // All the other parameters...
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
    // Physical location of y, x, z;
    float xPix = xStart + x*dx;
    float yPix = yStart + y*dy;
    float zPix = zStart + z*dz;
    float cInv = 1/speedOfSound;
    float apertureOrigin = (-nElements/2.0f + 0.5f)*pitch;

    for(int tx = 0; tx < nTx; ++tx) {
        pixelValue = 0.0f;
        pixelWeight = 0.0f;
        
        float txAngle = txAngles[tx];
        float txDistance = zPix*__cosf(txAngle) + xPix*__sinf(txAngle);
            
        txWeight = txApodization[IDX_3D(tx, x, z, nXPix, nZPix)];
        float elementPosition = apertureOrigin;
        if(txWeight == 1) {
            for(int rx = 0; rx < nRx; ++rx) {
                rxWeight = rxApodization[IDX_3D(rx, y, z, nYPix, nZPix)];
                if(rxWeight != 0.0f) {
                    // Distance of the (y, z) from the element.
                    float rxDistance = hypotf(yPix - (apertureOrigin + rx*pitch), zPix);
                    delay = (rxDistance+txDistance)*cInv;
                    delay = initDelay + delay;
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
