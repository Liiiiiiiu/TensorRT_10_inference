#include "yolov5_cu.h"
#include <fstream>
#include <cassert>
#include <iostream>
#include <chrono>
#include <random>
#include "postprocess.h"

#define MAX_IMAGE_INPUT_SIZE_THRESH 1920 * 1080 // 图片实际输入大小
#define MAX_OBJECTS 5000 // 增加到 1000 以支持更多目标
#define NUM_BOX_ELEMENT 15 // left, top, right, bottom, confidence, class, keepflag, landmarks[8]
#define NUM_CLASSES 3 // body, car, plate

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != cudaSuccess) { \
            std::cerr << "Cuda failure at " << __FILE__ << ":" << __LINE__ \
                      << ": " << ret << " (" << cudaGetErrorString(ret) << ")" << std::endl; \
            abort(); \
        } \
    } while (0)

static std::vector<std::string> yolov5_class_labels{"body", "car", "plate"};

// 定义计时器辅助类
class Timer {
public:
    Timer(const std::string& func_name) : func_name(func_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << func_name << " executed in " << duration << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string func_name;
};

void get_d2i_matrix(affine_matrix& afmt, cv::Size to, cv::Size from) {
    Timer timer("get_d2i_matrix");
    float scale = std::min(to.width / float(from.width), to.height / float(from.height));
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = -scale * from.width * 0.5 + to.width * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

    cv::Mat mat_i2d(2, 3, CV_32F, afmt.i2d);
    cv::Mat mat_d2i(2, 3, CV_32F, afmt.d2i);
    cv::invertAffineTransform(mat_i2d, mat_d2i);
    memcpy(afmt.d2i, mat_d2i.ptr<float>(0), sizeof(afmt.d2i));
}

yolov5_cu::yolov5_cu(const YAML::Node& config) : Model(config) {
    Timer timer("yolov5_cu::yolov5_cu");
    prob_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    output_candidates = 0; // Will be set in LoadEngine
    decode_ptr_host = nullptr;
    decode_ptr_device = nullptr;
}

yolov5_cu::~yolov5_cu() {
    Timer timer("yolov5_cu::~yolov5_cu");
    if (decode_ptr_host) {
        cudaFreeHost(decode_ptr_host);
        decode_ptr_host = nullptr;
    }
    if (decode_ptr_device) {
        CHECK(cudaFree(decode_ptr_device));
        decode_ptr_device = nullptr;
    }
}

void yolov5_cu::LoadEngine() {
    Timer timer("yolov5_cu::LoadEngine");

    // Call base class LoadEngine to initialize engine and context
    Model::LoadEngine();
    if (!engine || !context) {
        gLogError << "Failed to load engine or create context." << std::endl;
        return;
    }

    // Get output dimensions
    const char* outputTensorName = engine->getIOTensorName(1); // Assuming output is at index 1
    nvinfer1::Dims out_dims = engine->getTensorShape(outputTensorName);
    if (out_dims.nbDims != 3) {
        gLogError << "Invalid output dimensions for tensor " << outputTensorName
                  << ": expected 3 dims, got " << out_dims.nbDims << std::endl;
        return;
    }

    // Calculate output candidates (e.g., 25200 for 640x640 input)
    output_candidates = out_dims.d[1];
    int num_elements = out_dims.d[2];
    std::cout << "YOLOv5 output candidates: " << output_candidates
              << ", elements per candidate: " << num_elements << std::endl;

    // Verify output format
    if (num_elements != (5 + NUM_CLASSES + 8)) { // box(4) + score(1) + classes(3) + landmarks(8)
        gLogError << "Unexpected number of elements per candidate: " << num_elements
                  << ", expected " << (5 + NUM_CLASSES + 8) << std::endl;
        return;
    }

    // Allocate decode buffers
    if (decode_ptr_host) cudaFreeHost(decode_ptr_host);
    if (decode_ptr_device) cudaFree(decode_ptr_device);
    CHECK(cudaMallocHost(&decode_ptr_host, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    std::cout << "分配解码缓冲区：decode_ptr_host=" << decode_ptr_host
              << ", decode_ptr_device=" << decode_ptr_device
              << ", 大小=" << sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT) << " 字节" << std::endl;
}

std::vector<Yolov5Res> yolov5_cu::InferenceImages(std::vector<cv::Mat>& vec_img) {
    std::vector<Yolov5Res> yolov5_results_vec;
    if (vec_img.empty()) {
        gLogError << "输入图像向量为空。" << std::endl;
        return yolov5_results_vec;
    }

    // Assume batch size of 1
    cv::Mat img = vec_img[0];
    Yolov5Res yolov5_res;
    affine_matrix afmt;

    // Compute affine transformation matrix
    get_d2i_matrix(afmt, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), cv::Size(img.cols, img.rows));

    // Convert to RGB
    if (img.data) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    std::vector<cv::Mat> input_images = {img};

    // Preprocess
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<float> input_data = PreProcess(input_images);
    auto time_pre = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - begin_time).count();
    std::cout << "预处理时间: " << time_pre << " 毫秒" << std::endl;

    // Allocate GPU memory for output
    float* output_device = nullptr;
    CHECK(cudaMalloc(&output_device, outSize * sizeof(float)));
    std::cout << "分配输出设备内存：output_device=" << output_device << std::endl;

    // Inference with warmup
    auto infer_begin_time = std::chrono::high_resolution_clock::now();
    std::vector<float> output_data(outSize);
    // Warmup GPU
    for (int i = 0; i < 5; ++i) {
        ModelInference(input_data, output_data.data());
    }
    bool inference_success = ModelInference(input_data, output_data.data());
    // Copy inference output from buffers[1] to output_device
    CHECK(cudaMemcpyAsync(output_device, buffers[1], outSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    std::cout << "从 buffers[1] 复制推理输出到 output_device" << std::endl;
    auto time_infer = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - infer_begin_time).count();
    std::cout << "推理时间: " << time_infer << " 毫秒" << std::endl;

    if (!inference_success) {
        gLogError << "图像推理失败。" << std::endl;
        cudaFree(output_device);
        return yolov5_results_vec;
    }

    // Postprocess
    auto post_begin_time = std::chrono::high_resolution_clock::now();

    // Clear device buffer
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), stream));
    std::cout << "清空 decode_ptr_device 缓冲区" << std::endl;

    // Decode
    auto decode_begin_time = std::chrono::high_resolution_clock::now();
    decode_kernel_invoker(output_device, output_candidates, NUM_CLASSES, 4, prob_threshold,
                          afmt.d2i, decode_ptr_device, MAX_OBJECTS, stream);
    cudaError_t decode_err = cudaGetLastError();
    if (decode_err != cudaSuccess) {
        std::cerr << "解码内核启动失败: " << cudaGetErrorString(decode_err) << std::endl;
    } else {
        std::cout << "解码内核启动成功" << std::endl;
    }
    CHECK(cudaStreamSynchronize(stream)); // 确保内核完成
    auto time_decode = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - decode_begin_time).count();
    std::cout << "后处理-解码时间: " << time_decode << " 毫秒" << std::endl;
    
    // Check detection count before NMS
    float temp_count;
    CHECK(cudaMemcpyAsync(&temp_count, decode_ptr_device, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    std::cout << "解码后原始检测对象数量: " << (int)temp_count << std::endl;
    if ((int)temp_count > MAX_OBJECTS) {
        gLogError << "检测对象数量 " << (int)temp_count << " 超过 MAX_OBJECTS " << MAX_OBJECTS << std::endl;
        cudaFree(output_device);
        return yolov5_results_vec; // 提前返回
    }
    
    // NMS
    auto nms_begin_time = std::chrono::high_resolution_clock::now();
    nms_kernel_invoker(decode_ptr_device, nms_threshold, MAX_OBJECTS, stream);
    cudaError_t nms_err = cudaGetLastError();
    if (nms_err != cudaSuccess) {
        std::cerr << "NMS 内核启动失败: " << cudaGetErrorString(nms_err) << std::endl;
    } else {
        std::cout << "NMS 内核启动成功" << std::endl;
    }
    CHECK(cudaStreamSynchronize(stream)); // 确保 NMS 完成
    auto time_nms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - nms_begin_time).count();
    std::cout << "后处理-NMS 时间: " << time_nms << " 毫秒" << std::endl;
    
    // Verify pointers before copy
    if (!decode_ptr_host || !decode_ptr_device) {
        gLogError << "无效指针: decode_ptr_host=" << decode_ptr_host
                  << ", decode_ptr_device=" << decode_ptr_device << std::endl;
        cudaFree(output_device);
        return yolov5_results_vec;
    }
    std::cout << "复制前验证指针: decode_ptr_host=" << decode_ptr_host
              << ", decode_ptr_device=" << decode_ptr_device << std::endl;
    
    // Copy results
    auto copy_begin_time = std::chrono::high_resolution_clock::now();
    CHECK(cudaStreamSynchronize(stream)); // 确保前序操作完成
    cudaError_t copy_err = cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                          sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT),
                                          cudaMemcpyDeviceToHost, stream);
    if (copy_err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync 启动失败: " << cudaGetErrorString(copy_err) << std::endl;
    }
    CHECK(copy_err);
    CHECK(cudaStreamSynchronize(stream));
    auto time_copy = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - copy_begin_time).count();
    std::cout << "复制时间: " << time_copy << " 毫秒" << std::endl;

    // Print first few elements of decode_ptr_host for debugging
    std::cout << "decode_ptr_host 的前几个元素:" << std::endl;
    for (int i = 0; i < std::min(10, (int)(1 + MAX_OBJECTS * NUM_BOX_ELEMENT)); i++) {
        std::cout << "decode_ptr_host[" << i << "] = " << decode_ptr_host[i] << std::endl;
    }

    // Parse detection results
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    std::cout << "最终检测对象数量: " << count << std::endl;
    for (int i = 0; i < count; i++) {
        int basic_pos = 1 + i * NUM_BOX_ELEMENT;
        int keep_flag = (int)decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1) {
            Yolov5Box box;
            box.x1 = decode_ptr_host[basic_pos + 0];
            box.y1 = decode_ptr_host[basic_pos + 1];
            box.x2 = decode_ptr_host[basic_pos + 2];
            box.y2 = decode_ptr_host[basic_pos + 3];
            box.score = decode_ptr_host[basic_pos + 4];
            box.label = (int)decode_ptr_host[basic_pos + 5];
            int landmark_pos = basic_pos + 7;
            for (int id = 0; id < 4; id++) {
                box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
            }
            yolov5_res.yolov5_results.push_back(box);
            std::cout << "检测框 " << i << ": 类别=" << yolov5_class_labels[box.label]
                      << ", 置信度=" << box.score << ", 边界框=(" << box.x1 << "," << box.y1
                      << "," << box.x2 << "," << box.y2 << ")" << std::endl;
        }
    }
    yolov5_results_vec.push_back(yolov5_res);

    auto time_post = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - post_begin_time).count();
    std::cout << "后处理总时间: " << time_post << " 毫秒" << std::endl;

    // Convert back to BGR for saving
    if (img.data) {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }
    input_images[0] = img;

    // Free GPU memory
    cudaFree(output_device);

    return yolov5_results_vec;
}

void yolov5_cu::InferenceFolder(const std::string& folder_name) {
    Timer timer("yolov5_cu::InferenceFolder");

    std::vector<std::string> image_list = ReadFolder(folder_name);
    int index = 0;
    int batch_id = 0;
    int batch_size = 1; // Assuming batch size of 1
    std::vector<cv::Mat> vec_Mat(batch_size);
    std::vector<std::string> vec_name(batch_size);
    float total_time = 0;

    for (const std::string& image_name : image_list) {
        // Skip result images
        if (image_name.find("_result") != std::string::npos) {
            std::cout << "跳过结果图像: " << image_name << std::endl;
            continue;
        }
        index++;
        std::cout << "处理图像: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == batch_size || index == image_list.size()) {
            vec_Mat.resize(batch_id);
            vec_name.resize(batch_id);

            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();

            if (!det_results.empty()) {
                DrawResults(det_results, vec_Mat, vec_name);
            } else {
                gLogWarning << "批次 " << vec_name[0] << " 无有效结果" << std::endl;
            }

            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
            vec_Mat = std::vector<cv::Mat>(batch_size);
            vec_name = std::vector<std::string>(batch_size);
            batch_id = 0;
        }
    }
    if (!image_list.empty()) {
        std::cout << "平均处理时间: " << total_time / image_list.size() << " 毫秒" << std::endl;
    }
}

void yolov5_cu::DrawResults(const std::vector<Yolov5Res>& detections, std::vector<cv::Mat>& vec_img,
                            std::vector<std::string>& image_name) {
    Timer timer("yolov5_cu::DrawResults");

    std::vector<cv::Scalar> class_colors(NUM_CLASSES);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);
    for (cv::Scalar& class_color : class_colors) {
        class_color = cv::Scalar(dist(rng), dist(rng), dist(rng));
    }

    for (size_t i = 0; i < vec_img.size(); i++) {
        auto& org_img = vec_img[i];
        if (!org_img.data) continue;

        if (i >= detections.size()) {
            gLogWarning << "图像索引 " << i << " 无检测结果" << std::endl;
            continue;
        }

        const auto& rects = detections[i].yolov5_results;
        for (const auto& rect : rects) {
            char t[256];
            snprintf(t, sizeof(t), "%.2f", rect.score);
            std::string name = yolov5_class_labels[rect.label] + "-" + t;
            cv::putText(org_img, name, cv::Point(rect.x1, rect.y1 - 5),
                        cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.label], 2);
            cv::Rect rst(cv::Point(rect.x1, rect.y1), cv::Point(rect.x2, rect.y2));
            cv::rectangle(org_img, rst, class_colors[rect.label], 2, cv::LINE_8, 0);
            // Draw landmarks
            for (int k = 0; k < 4; k++) {
                cv::circle(org_img, cv::Point(rect.landmarks[2 * k], rect.landmarks[2 * k + 1]),
                           3, cv::Scalar(0, 255, 0), -1);
            }
        }

        if (!image_name.empty() && i < image_name.size()) {
            std::string rst_name = image_name[i];
            size_t pos = rst_name.find_last_of('.');
            if (pos != std::string::npos) {
                rst_name.insert(pos, "_result");
            } else {
                rst_name += "_result.jpg";
            }
            std::cout << "保存结果到: " << rst_name << std::endl;
            cv::imwrite(rst_name, org_img);
        }
    }
}