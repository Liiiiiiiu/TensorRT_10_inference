//
// Created by linghu8812 on 2021/2/8.
//

#include "model.h"
#include <NvInferPlugin.h>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Logger instance (assumed defined in common.h)
#include "common.h"

// Model constructor
Model::Model(const YAML::Node& config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    image_order = config["image_order"].as<std::string>();
    channel_order = config["channel_order"].as<std::string>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_std"].as<std::vector<float>>();
    alpha = config["alpha"].as<float>();
    resize = config["resize"].as<std::string>();
    buffers.resize(2, nullptr); // Initialize for input/output bindings
}

// Model destructor
Model::~Model() {
    for (auto& buffer : buffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    std::cout << "Model destroyed." << std::endl;
}

void Model::OnnxToTRTModel() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        gLogError << "createInferBuilder failed" << std::endl;
        return;
    }

    // TensorRT 10.x 默认支持显式批处理，无需 kEXPLICIT_BATCH
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
        gLogError << "createNetworkV2 failed" << std::endl;
        return;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        gLogError << "createBuilderConfig failed" << std::endl;
        return;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        gLogError << "createParser failed" << std::endl;
        return;
    }

    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        gLogError << "Failure while parsing ONNX file: " << onnx_file << std::endl;
        return;
    }

    // Set optimization profile for dynamic shapes
    auto profile = builder->createOptimizationProfile();
    auto input_name = network->getInput(0)->getName();
    nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
    input_dims.d[0] = BATCH_SIZE; // Fix batch size
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    // Configure builder
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 8ULL << 30); // 8 GiB
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled." << std::endl;
    }

    std::cout << "Building engine..." << std::endl;
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        gLogError << "buildSerializedNetwork failed" << std::endl;
        return;
    }
    std::cout << "Engine built successfully." << std::endl;

    // Save engine to file
    std::ofstream file(engine_file, std::ios::binary);
    if (!file.is_open()) {
        gLogError << "Failed to open engine file: " << engine_file << std::endl;
        return;
    }
    file.write(static_cast<const char*>(plan->data()), plan->size());
    file.close();
    std::cout << "Engine saved to: " << engine_file << std::endl;
}

bool Model::ReadTrtFile() {
    std::ifstream file(engine_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        gLogError << "Error opening engine file: " << engine_file << std::endl;
        return false;
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and store it
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    if (!runtime) {
        gLogError << "createInferRuntime failed" << std::endl;
        return false;
    }

    // Deserialize engine
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) {
        gLogError << "deserializeCudaEngine failed" << std::endl;
        return false;
    }
    std::cout << "Engine deserialized successfully." << std::endl;
    return true;
}

void Model::LoadEngine() {
    std::cout << "Using TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    std::ifstream file_check(engine_file, std::ios::binary);
    if (file_check.good()) {
        file_check.close();
        if (!ReadTrtFile()) {
            gLogError << "Failed to read engine file. Building from ONNX." << std::endl;
            OnnxToTRTModel();
            if (!ReadTrtFile()) {
                gLogError << "Failed to load newly built engine." << std::endl;
                return;
            }
        }
    } else {
        file_check.close();
        std::cout << "Engine file not found. Building from ONNX: " << onnx_file << std::endl;
        OnnxToTRTModel();
        if (!ReadTrtFile()) {
            gLogError << "Failed to load newly built engine." << std::endl;
            return;
        }
    }

    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        gLogError << "createExecutionContext failed" << std::endl;
        return;
    }

    // Allocate buffers for all I/O tensors
    int nbTensors = engine->getNbIOTensors();
    buffers.resize(nbTensors, nullptr);
    bufferSize.resize(nbTensors, 0);
    for (int i = 0; i < nbTensors; ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(tensorName);
        if (dims.nbDims == -1) {
            gLogError << "Tensor " << tensorName << " has invalid dimensions." << std::endl;
            return;
        }
        nvinfer1::DataType dtype = engine->getTensorDataType(tensorName);
        bufferSize[i] = volume(dims) * getElementSize(dtype);
        std::cout << "Allocating buffer for tensor " << tensorName << ": " << bufferSize[i] << " bytes" << std::endl;
        if (buffers[i]) cudaFree(buffers[i]);
        if (cudaMalloc(&buffers[i], bufferSize[i]) != cudaSuccess) {
            gLogError << "CUDA malloc failed for tensor " << tensorName << std::endl;
            return;
        }
    }

    // Create CUDA stream
    if (stream) cudaStreamDestroy(stream);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        gLogError << "cudaStreamCreate failed" << std::endl;
        return;
    }

    outSize = bufferSize[1] / sizeof(float); // Assuming output is float
    std::cout << "Output size: " << outSize << " floats" << std::endl;
}

std::vector<float> Model::PreProcess(std::vector<cv::Mat>& vec_img) {
    std::vector<float> result(BATCH_SIZE * INPUT_CHANNEL * IMAGE_WIDTH * IMAGE_HEIGHT);
    float* data = result.data();
    int single_image_size = INPUT_CHANNEL * IMAGE_WIDTH * IMAGE_HEIGHT;

    for (int b = 0; b < vec_img.size(); ++b) {
        auto& img = vec_img[b];
        float* img_data = data + b * single_image_size;

        if (img.empty()) {
            gLogWarning << "Image " << b << " is empty." << std::endl;
            std::fill(img_data, img_data + single_image_size, 0.0f);
            continue;
        }

        // Assume input is RGB (converted in InferenceFolder)
        cv::Mat resized_img;
        if (resize == "directly") {
            cv::resize(img, resized_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        } else if (resize == "keep_ratio") {
            float ratio = std::min(static_cast<float>(IMAGE_WIDTH) / img.cols,
                                   static_cast<float>(IMAGE_HEIGHT) / img.rows);
            cv::Mat tmp;
            cv::resize(img, tmp, cv::Size(), ratio, ratio);
            resized_img = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, img.type());
            tmp.copyTo(resized_img(cv::Rect(0, 0, tmp.cols, tmp.rows)));
        } else {
            gLogError << "Unknown resize mode: " << resize << std::endl;
            resized_img = img;
        }

        resized_img.convertTo(resized_img, CV_32F, 1.0 / alpha);
        std::vector<cv::Mat> channels(INPUT_CHANNEL);
        cv::split(resized_img, channels);

        int channel_size = IMAGE_WIDTH * IMAGE_HEIGHT;
        if (image_order == "BCHW") {
            for (int c = 0; c < INPUT_CHANNEL; ++c) {
                if (img_mean.size() == INPUT_CHANNEL && img_std.size() == INPUT_CHANNEL) {
                    channels[c] = (channels[c] - img_mean[c]) / img_std[c];
                }
                memcpy(img_data + c * channel_size, channels[c].data, channel_size * sizeof(float));
            }
        } else if (image_order == "BHWC") {
            if (img_mean.size() == INPUT_CHANNEL && img_std.size() == INPUT_CHANNEL) {
                for (int c = 0; c < INPUT_CHANNEL; ++c) {
                    channels[c] = (channels[c] - img_mean[c]) / img_std[c];
                }
            }
            cv::Mat merged;
            cv::merge(channels, merged);
            memcpy(img_data, merged.data, single_image_size * sizeof(float));
        } else {
            gLogError << "Unknown image order: " << image_order << std::endl;
        }
    }
    return result;
}
bool Model::ModelInference(std::vector<float> image_data, float* output) {
    if (!context || !engine) {
        gLogError << "Engine or context not initialized." << std::endl;
        return false;
    }
    if (image_data.empty() || !output) {
        gLogError << "Invalid input or output data." << std::endl;
        return false;
    }

    // Assuming input tensor is at index 0, output at index 1
    const char* inputTensorName = engine->getIOTensorName(0);
    const char* outputTensorName = engine->getIOTensorName(1);

    // Copy input data to device
    if (cudaMemcpyAsync(buffers[0], image_data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "CUDA memcpy to device failed for tensor " << inputTensorName << std::endl;
        return false;
    }

    // Set tensor addresses
    if (!context->setTensorAddress(inputTensorName, buffers[0])) {
        gLogError << "Failed to set tensor address for " << inputTensorName << std::endl;
        return false;
    }
    if (!context->setTensorAddress(outputTensorName, buffers[1])) {
        gLogError << "Failed to set tensor address for " << outputTensorName << std::endl;
        return false;
    }

    // Set input dimensions
    if (!context->setInputShape(inputTensorName, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH})) {
        gLogError << "Failed to set input tensor dimensions for " << inputTensorName << std::endl;
        return false;
    }

    // Execute inference
    if (!context->enqueueV3(stream)) {
        gLogError << "enqueueV3 failed." << std::endl;
        return false;
    }

    // Copy output
    if (cudaMemcpyAsync(output, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        gLogError << "CUDA memcpy to host failed for tensor " << outputTensorName << std::endl;
        return false;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        gLogError << "CUDA stream synchronization failed." << std::endl;
        return false;
    }

    return true;
}

int64_t Model::volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i] > 0 ? dims.d[i] : 1;
    }
    return v;
}

int Model::getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kFP8: return 1;
        case nvinfer1::DataType::kBF16: return 2;
        case nvinfer1::DataType::kINT64: return 8;
        default: throw std::runtime_error("Unsupported data type");
    }
}