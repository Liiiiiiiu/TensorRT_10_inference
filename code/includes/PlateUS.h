#pragma once
#include "classification.h"

struct PlateResult {
    std::string plate_label;
    ClassRes base_result;
};

class PlateUS : public Classification {
public:
    PlateUS(const YAML::Node &config);
    std::vector<ClassRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
    std::vector<PlateResult> PostProcessDetailed(const std::vector<cv::Mat> &vec_Mat, float *output);
    std::string decodePlate(const std::vector<int>& preds);

private:
    std::string plate_chr = "#0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
};
