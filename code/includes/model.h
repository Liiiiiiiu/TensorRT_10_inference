#ifndef TENSORRT_INFERENCE_MODEL_H
#define TENSORRT_INFERENCE_MODEL_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "common.h"
#include <yaml-cpp/yaml.h>
#include <memory>
#include <vector>
#include <string>

class Model {
public:
    explicit Model(const YAML::Node& config);
    virtual ~Model();
    virtual void LoadEngine();
    virtual void InferenceFolder(const std::string& folder_name) = 0;

protected:
    bool ReadTrtFile();
    void OnnxToTRTModel();
    virtual std::vector<float> PreProcess(std::vector<cv::Mat>& image);
    bool ModelInference(std::vector<float> image_data, float* output); // 修改为返回 bool

    // Helper functions
    int64_t volume(const nvinfer1::Dims& dims);
    int getElementSize(nvinfer1::DataType t);

    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    std::unique_ptr<nvinfer1::IRuntime> runtime; // 新增：存储 IRuntime
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void*> buffers; // Dynamic buffer array
    std::vector<int64_t> bufferSize;
    cudaStream_t stream = nullptr;
    int outSize;
    std::string image_order;
    std::string channel_order;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    float alpha;
    std::string resize;
};

#endif // TENSORRT_INFERENCE_MODEL_H