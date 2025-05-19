#ifndef TENSORRT_INFERENCE_YOLOV8POSE_H
#define TENSORRT_INFERENCE_YOLOV8POSE_H

#include "classification.h"
//#include "opencv2/opencv.hpp"

struct Bbox1 : ClassRes{
    float x;
    float y;
    float w;
    float h;
};

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};
struct DetectRes1 {
    std::vector<Bbox1> det_results;
    std::vector<std::vector<float>> det_kpss;
};

class YOLOv8Pose : public Model
{
public:
    explicit YOLOv8Pose(const YAML::Node &config);
    std::vector<DetectRes1> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<DetectRes1> &detections, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_name);
 

protected:
    virtual std::vector<DetectRes1> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    std::map<int, std::string> class_labels;
    int CATEGORY;
    float obj_threshold;
    float nms_threshold;
    bool agnostic;
    std::vector<cv::Scalar> class_colors;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    int num_rows = 0;
    
    
	int m_output_objects_width;

	int m_nkpts;
	
};



#endif //TENSORRT_INFERENCE_YOLOV8POSE_H
