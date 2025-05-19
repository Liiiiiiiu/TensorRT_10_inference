#ifndef TENSORRT_INFERENCE_RESNET_H
#define TENSORRT_INFERENCE_RESNET_H

#include "classification.h"

class Resnet : public Classification {
public:
    explicit Resnet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_RESNET_H
