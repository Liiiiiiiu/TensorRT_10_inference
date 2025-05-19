#include "classification.h"

Classification::Classification(const YAML::Node &config) : Model(config) {
    labels_file = config["labels_file"].as<std::string>();
    class_labels = ReadImageNetLabel(labels_file);
    CATEGORY = class_labels.size();
}

std::vector<ClassRes> Classification::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "classification prepare image take: " << total_pre << " ms." << std::endl;

    std::vector<float> output(outSize * BATCH_SIZE); // 使用 vector 管理内存
    auto t_start = std::chrono::high_resolution_clock::now();
    bool inference_success = ModelInference(image_data, output.data());
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "classification inference take: " << total_inf << " ms." << std::endl;

    std::vector<ClassRes> results;
    if (!inference_success) {
        gLogError << "Inference failed, skipping postprocessing." << std::endl;
        return results; // 返回空结果
    }

    auto r_start = std::chrono::high_resolution_clock::now();
    results = PostProcess(vec_img, output.data());
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "classification postprocess take: " << total_res << " ms." << std::endl;

    return results;
}

void Classification::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> image_list = ReadFolder(folder_name);
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    std::vector<std::string> vec_name(BATCH_SIZE);
    float total_time = 0;

    for (const std::string &image_name : image_list) {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {
            // Convert BGR to RGB for model input
            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == BATCH_SIZE || index == image_list.size()) {
            // Trim vec_Mat and vec_name to actual size
            vec_Mat.resize(batch_id);
            vec_name.resize(batch_id);

            auto start_time = std::chrono::high_resolution_clock::now();
            auto cls_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();

            // Convert back to BGR for saving
            for (auto& img : vec_Mat) {
                if (img.data) {
                    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                }
            }

            if (!cls_results.empty()) {
                DrawResults(cls_results, vec_Mat, vec_name);
            } else {
                gLogWarning << "No valid results for batch starting with " << vec_name[0] << std::endl;
            }

            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            vec_name = std::vector<std::string>(BATCH_SIZE);
            batch_id = 0;
        }
    }

    if (!image_list.empty()) {
        std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
    }
}

std::vector<ClassRes> Classification::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<ClassRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        ClassRes result;
        float *out = output + index * outSize;
        // Apply softmax
        float max_val = *std::max_element(out, out + outSize); // Subtract max for numerical stability
        float sum = 0.0f;
        std::vector<float> exp_scores(outSize);
        for (int i = 0; i < outSize; ++i) {
            exp_scores[i] = std::exp(out[i] - max_val);
            sum += exp_scores[i];
        }
        auto max_pos = std::max_element(exp_scores.begin(), exp_scores.end());
        result.classes = max_pos - exp_scores.begin();
        result.prob = *max_pos / sum;
        std::cout << "Image " << index << ": Class " << result.classes << " (" << class_labels[result.classes]
                  << "), Probability " << result.prob << std::endl;
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void Classification::DrawResults(const std::vector<ClassRes> &results, std::vector<cv::Mat> &vec_img,
    std::vector<std::string> image_names) {
for (size_t i = 0; i < vec_img.size(); ++i) {
auto& org_img = vec_img[i];
if (!org_img.data) continue;

if (i >= results.size()) {
gLogWarning << "No result for image index " << i << std::endl;
continue;
}

const auto& result = results[i];
std::string label = class_labels[result.classes] + ": " + std::to_string(result.prob);
cv::putText(org_img, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

if (!image_names.empty() && i < image_names.size()) {
std::string rst_name = image_names[i];
size_t pos = rst_name.find_last_of('.');
if (pos != std::string::npos) {
rst_name.insert(pos, "_result");
} else {
rst_name += "_result.png"; // 统一为 .png
}
std::cout << "Saving result to: " << rst_name << std::endl;
cv::imwrite(rst_name, org_img);
}
}
}
