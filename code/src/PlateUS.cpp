#include "PlateUS.h"

PlateUS::PlateUS(const YAML::Node &config) : Classification(config) {}

std::vector<ClassRes> PlateUS::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    // Get detailed results with additional plate recognition information
    auto detailed_results = PostProcessDetailed(vec_Mat, output);
    std::vector<ClassRes> vec_result;
    vec_result.reserve(detailed_results.size());  // Pre-allocate space for efficiency

    // Extract base results from detailed results
    for (const auto& detailed : detailed_results) {
        vec_result.push_back(detailed.base_result);
    }
    return vec_result;
}

// Detailed processing for plate recognition results
std::vector<PlateResult> PlateUS::PostProcessDetailed(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<PlateResult> vec_result;
    vec_result.reserve(vec_Mat.size());  // Pre-allocate space for expected results
    int index = 0;

    for (const auto &src_img : vec_Mat) {
        PlateResult result;
        float *out = output + index * 21 * 35;  // Offset pointer for each image's result data

        std::vector<int> preds(21);

        // For each character position, find the index of the highest probability character
        for (int i = 0; i < 21; ++i) {
            float *current_char = out + i * 35;
            preds[i] = std::max_element(current_char, current_char + 35) - current_char;
        }

        // Decode the plate result from predicted character indices
        result.plate_label = decodePlate(preds);
        std::cout << "License Plate Recognition Result: " << result.plate_label << std::endl;

        vec_result.push_back(result);
        ++index;
    }

    return vec_result;
}

// Decodes character indices into a license plate string
std::string PlateUS::decodePlate(const std::vector<int> &preds) {
    std::vector<int> newPreds;
    int pre = -1;  // Initialize to avoid matching the first character

    // Remove invalid (0) and consecutive duplicate characters
    for (int pred : preds) {
        if (pred != 0 && pred != pre) {
            newPreds.push_back(pred);
        }
        pre = pred;
    }

    // Convert character indices to the actual plate characters
    std::string plate;
    for (int i : newPreds) {
        plate += plate_chr[i];
    }

    return plate;
}
