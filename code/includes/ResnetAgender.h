#pragma once
#include "classification.h"
#include <string>

struct AgenderResult{
    std::string age_label;
    std::string gender_label;
    float gender_prob;
    ClassRes base_result; // containing general classification data 
};

class ResnetAgender : public Classification{

public:

    ResnetAgender(const YAML::Node &config);

    std::vector<ClassRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;

    std::vector<AgenderResult> PostProcessDetailed(const std::vector<cv::Mat> &vec_Mat, float *output);
};
