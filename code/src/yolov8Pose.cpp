#include "yolov8Pose.h"

//#include "opencv2/opencv.hpp"


YOLOv8Pose::YOLOv8Pose(const YAML::Node &config) : Model(config) {

 labels_file = config["labels_file"].as<std::string>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    agnostic = config["agnostic"].as<bool>();
    strides = config["strides"].as<std::vector<int>>();
    num_anchors = config["num_anchors"].as<std::vector<int>>();
    int index = 0;
    for (const int &stride : strides)
    {
        int num_anchor = num_anchors[index] !=0 ? num_anchors[index] : 1;
        num_rows += int(IMAGE_HEIGHT / stride) * int(IMAGE_WIDTH / stride) * num_anchor;
        index+=1;
    }
    class_labels = ReadClassLabel(labels_file);
    CATEGORY = class_labels.size();
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        
    m_nkpts = 17;
    m_output_objects_width = 56; // xywhc + points * 17 = 56  -> left, top, right, bottom, confidence, class, keepflag + points * 17 = 58
    
}


std::vector<DetectRes1> YOLOv8Pose::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "detection prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    std::cout << "detection outSize: " << outSize << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "detection inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto boxes = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "detection postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return boxes;
}

void YOLOv8Pose::InferenceFolder(const std::string &folder_name) {
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
            if (channel_order == "BGR")
                cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            DrawResults(det_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}


void YOLOv8Pose::DrawResults(const std::vector<DetectRes1> &detections, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> image_name=std::vector<std::string>()) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto rects = detections[i].det_results;
        if (channel_order == "BGR")
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for(const auto &rect : rects) {
            char t[256];
            sprintf(t, "%.2f", rect.prob);
            std::string name = class_labels[rect.classes] + "-" + t;
            cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.classes], 2);
            cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
            cv::rectangle(org_img, rst, class_colors[rect.classes], 2, cv::LINE_8, 0);
        }
        
        auto kpss = detections[i].det_kpss;
    
        for(const auto &kps : kpss) {
        
           for (int k = 0; k < m_nkpts + 2; k++) {
            if (k < m_nkpts) {
                int   kps_x = std::round(kps[k * 3]);
                int   kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(org_img, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto& ske    = SKELETON[k];
            int   pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int   pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(org_img, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
        
           
        }
        
        
        if (!image_name.empty()) {
            int pos = image_name[i].find_last_of('.');
            std::string rst_name = image_name[i].insert(pos, "_");
            std::cout << rst_name << std::endl;
            cv::imwrite(rst_name, org_img);
        }
    }
}

std::vector<DetectRes1> YOLOv8Pose::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<DetectRes1> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        
        
        std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;
        float height = float(src_img.rows);
        float width = float(src_img.cols);
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        std::cout << "src_img.cols:" << src_img.cols << std::endl;
        std::cout << "IMAGE_WIDTH:" << IMAGE_WIDTH << std::endl;
        std::cout << "src_img.rows:" << src_img.rows << std::endl;
        std::cout << "IMAGE_HEIGHT:" << IMAGE_HEIGHT << std::endl;
        std::cout << "ratio:" << ratio << std::endl;
        std::cout << "outSize:" << outSize << std::endl;
        std::cout << "CATEGORY:" << CATEGORY << std::endl;
        std::cout << "num_rows:" << num_rows << std::endl;
        float *out = output + index * outSize;
        cv::Mat res_mat = cv::Mat(m_output_objects_width, num_rows, CV_32FC1, out);
//        std::cout << res_mat << std::endl;
        res_mat = res_mat.t();
//        out = res_mat.ptr<float>(0);
        //cv::Mat prob_mat;
        //cv::reduce(res_mat.colRange(4, CATEGORY + 4), prob_mat, 1, cv::REDUCE_MAX);
        //out = res_mat.ptr<float>(0);
        for (int position = 0; position < num_rows; position++) {
         auto row_ptr =res_mat.row(position).ptr<float>();
         
         auto bboxes_ptr = row_ptr;
         auto scores_ptr = row_ptr + 4;
         auto kps_ptr    = row_ptr + 5;
        
        
           cv::Rect_<float> bbox;
            //box.prob = *prob_mat.ptr<float>(position);
           float score = *scores_ptr;
           
           
           if (score > obj_threshold) {
        
           
              bbox.x = bboxes_ptr[0] * ratio;
              bbox.y = bboxes_ptr[1] * ratio;
              bbox.width = bboxes_ptr[2] * ratio;
              bbox.height = bboxes_ptr[3] * ratio;
           
              std::vector<float> kps;
              for (int k = 0; k < m_nkpts; k++) {
                float kps_x = (*(kps_ptr + 3 * k)) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1)) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
     
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
               }
            bboxes.push_back(bbox);
            labels.push_back(0);
            scores.push_back(score);
            kpss.push_back(kps);
           }
        }
        
        cv::dnn::NMSBoxes(bboxes, scores, obj_threshold, nms_threshold, indices);
        int cnt = 0;
        int topk = 1000;
        DetectRes1 result;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Bbox1 box;
        box.classes = labels[i];
            box.x = bboxes[i].x;
            box.y = bboxes[i].y;
            box.w = bboxes[i].width;
            box.h = bboxes[i].height;
            box.prob =scores[i];
            
        result.det_results.push_back(box);
        result.det_kpss.push_back( kpss[i]);
        cnt += 1;
    }
            
        //NmsDetect(result.det_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}



