#include "ResnetAgender.h"

ResnetAgender::ResnetAgender(const YAML::Node &config) : Classification(config) {}

std::vector<ClassRes> ResnetAgender::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    // Calls detailed post-processing and extracts base classification results
    auto detailed_results = PostProcessDetailed(vec_Mat, output);
    std::vector<ClassRes> vec_result;
    vec_result.reserve(detailed_results.size());  // Reserve space for efficiency

    for (const auto& detailed : detailed_results) {
        vec_result.push_back(detailed.base_result);
    }
    return vec_result;
}

// Processes output for detailed age and gender results
std::vector<AgenderResult> ResnetAgender::PostProcessDetailed(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<AgenderResult> vec_result;
    vec_result.reserve(vec_Mat.size());  // Reserve space for expected results
    int index = 0;

    for (const auto &src_img : vec_Mat) {
        AgenderResult result;
        float *out = output + index * 7;  // Offset pointer for each image's result

        // Print raw model output for debugging
        std::cout << "raw out: [";
        for (int i = 0; i < 7; ++i) {
            std::cout << out[i] << ((i != 6) ? " " : "");
        }
        std::cout << "]" << std::endl;

        // Age classification based on maximum probability across age classes
        auto max_pos = std::max_element(out, out + 6);
        result.base_result.classes = std::distance(out, max_pos);
        result.base_result.prob = *max_pos;

        // Map class index to age ranges
        if (result.base_result.classes <= 1) {
            result.age_label = "0-20";
        } else if (result.base_result.classes <= 3) {
            result.age_label = "21-40";
        } else if (result.base_result.classes == 4) {
            result.age_label = "41-60";
        } else {
            result.age_label = ">60";
        }

        // Gender classification
        float gender_prob = out[6];
        result.gender_label = (gender_prob >= 0.5) ? "male" : "female";
        result.gender_prob = (gender_prob >= 0.5) ? gender_prob : (1.0f - gender_prob);

        // Print final age and gender results for debugging
        std::cout << "{'gender': '" << result.gender_label << "', "
                  << "'gender_confident': " << result.gender_prob << ", "
                  << "'age': '" << result.age_label << "', "
                  << "'age_confident': " << result.base_result.prob << "}" << std::endl;

        vec_result.push_back(result);
        ++index;
    }

    return vec_result;
}
