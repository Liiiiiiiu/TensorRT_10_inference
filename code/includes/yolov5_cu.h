#ifndef __YOLOV5_CU_H
#define __YOLOV5_CU_H

#include "model.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "common.h" // Include shared definitions

struct Yolov5Box {
    float x1, y1, x2, y2; // Bounding box coordinates
    float score;          // Confidence score
    int label;            // Class label
    float landmarks[8];   // 4 keypoints (x1, y1, x2, y2, x3, y3, x4, y4)
};

struct Yolov5Res {
    std::vector<Yolov5Box> yolov5_results;
};

class yolov5_cu : public Model {
public:
    yolov5_cu(const YAML::Node& config);
    ~yolov5_cu();
    void LoadEngine() override;
    std::vector<Yolov5Res> InferenceImages(std::vector<cv::Mat>& vec_img);
    void InferenceFolder(const std::string& folder_name);

protected:
    void DrawResults(const std::vector<Yolov5Res>& detections,
                     std::vector<cv::Mat>& vec_img,
                     std::vector<std::string>& image_name);

private:
    float prob_threshold;
    float nms_threshold;
    int output_candidates;
    float* decode_ptr_host;
    float* decode_ptr_device;
};

#endif // __YOLOV5_CU_H