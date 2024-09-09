// Created by Hassan Nahas @ LITMUS based on SampleONNXMNIST by NVIDIA.
//
// Shows how TensorRT may be used to load an ONNX model for inference on the GPU on sample MNIST data.
// The model is loaded from Cpp\sample_mnist_data\
// 
// Specify parameters in the beginning of file.
//
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <chrono>
using namespace std::chrono;

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

// Parameters for File
// Path to ONNX model file
std::string onnxModelPath = "../sample_mnist_data/simple_nn.onnx";
// Path to MNIST dataset for testing
std::string datasetPath = "../sample_mnist_data/";
// Name of input layer from tensorflow
std::string inputTensorName = "input_1";
// Input of output layer form tensorflow
std::string outputTensorName = "softmax";
// Number of repeats for inference averaging
int numTrials = 20;
// For FP16 model compilation
bool fp16 = false;
// For int8 model compilation
bool int8 = false;
// Number of DLA cores to use (needs to supprting device)
int dlaCore = -1; // Can be 0 to n-1 for devices that support DLAs

nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
int mNumber{ 0 };             //!< The number to classify

std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    sample::gLogInfo << "inputH:" << inputH << std::endl;
    sample::gLogInfo << "inputW:" << inputW << std::endl;
    mNumber = rand() % 10;
    //readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", datasetPath), fileData.data(), inputH, inputW);
    char pgmPath[256];
    sprintf(pgmPath, "%s\%s.pgm", datasetPath.c_str(), std::to_string(mNumber));
    sample::gLogInfo << "pgmPath:" << pgmPath << std::endl;
    readPGMFile(pgmPath, fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(outputTensorName));
    float val{0.0F};
    int idx{0};

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        //output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9F;
}


int main(int argc, char** argv)
{
    sample::gLogInfo << "Parsing ONNX model to create TRT inference engine" << std::endl;
    // create builder object
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // create network object
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // create config object
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // create parser object
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    // parse ONNX model from filepath
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    // Enable FP16 or Int8 builder configurations
    if (fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    // Configure DLA
    samplesCommon::enableDLA(builder.get(), config.get(), dlaCore);


    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Build serialized network and return it
    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
        return false;
    }

    // Get inference runtime
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    // Get inference engine for network
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // Get dimensions of network
    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);
    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);
    
    // buffers for input/output of model
    samplesCommon::BufferManager buffers(mEngine);

    // Create execution context for inference from engine.
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    // Run inference once (for warming)
    bool status = context->executeV2(buffers.getDeviceBindings().data()); // synronous inference on GPU
    if (!status)
    {
        return false;
    }
    // Run inference numTrials times for averaging
    auto start = high_resolution_clock::now();
    for (int i = 0; i < numTrials; i++)
    {
        bool status = context->executeV2(buffers.getDeviceBindings().data()); // synronous inference on GPU
        if (!status)
        {
            return false;
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    sample::gLogInfo << "Mean inference over " << numTrials << " samples on the GPU = " << (float) (duration.count()) / 1000.0f / (float) (numTrials) << " ms" << std::endl;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;

}
