## 1. according the tensorrt_inference INSTALL.md to install TensorRT, OpenCV, yaml-cpp 

(1)IF tensorRT and Opencv version are not same as INSTALL.md. They are also surpport.

Verify tensorRT and Opencv version are: 
TensorRT-8.4.1.5
opencv-4.5.5


(2) **yaml-cpp 0.6.3**
- [download](https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.3.zip)
- Build static library
```
mkdir build && cd build
cmake ..
make -j
```
need to add -fPIC into ./yaml-cpp-yaml-cpp-0.6.3/CMakeLists.txt

in CMakeLists.txt after
project(YAML_CPP VERSION 0.6.3)
add
add_compile_options(-fPIC)


after make
copy the new libyaml-cpp.a to ./tensorrt_inference/depends/yaml-cpp/libs/libyaml-cpp.a

##2. change cmake to include tensorrt and make tensorrt-inference

(1) change ./tensorrt_inference/project/CMakeLists.txt.

A) add tensorRT path, after set(CMAKE_CXX_STANDARD 14)

set(TENSORRT_INCLUDE_DIR /home/linkdome/TensorRT-8.4.1.5/include)
set(TENSORRT_LIBRARIES /home/linkdome/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib)
set(TENSORRT_LIBRARY_INFER /home/linkdome/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib)
set(TENSORRT_LIBRARY_ONNXPARSER /home/linkdome/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib)
set(TENSORRT_LIBRARY_PLUGIN /home/linkdome/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib)

B) add TENSORRT_LIBRARIES into linkd_directories

link_directories(${YAML_LIB_DIR} ${TENSORRT_LIBRARIES})

C) add pthread nvinfer nvonnxparser nvinfer_plugin into target_link_libraries
target_link_libraries(tensorrt_inference yaml-cpp factory alexnet arcface CenterFace efficientnet face_alignment
        fast-reid FCN gender-age ghostnet lenet MiniFASNet mmpose nanodet RetinaFace ScaledYOLOv4 scrfd seresnext
        Swin-Transformer yolor Yolov4 yolov5 YOLOv6 yolov7 yolov8 pthread nvinfer nvonnxparser nvinfer_plugin)
        
(2) make tensorrt_inference   
cd tensorrt_inference/project
mkdir build && cd build
cmake ..
make -j


if make meet the NvInfer.h is not include. 
Please copy all tensorRT include h files to ./tensorrt_inference/code/includes


## 3. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv) or export onnx:
```bash
# 🔥 yolov8 offical repo: https://github.com/ultralytics/ultralytics
# 🔥 yolov8 quickstart: https://docs.ultralytics.com/quickstart/
# 🚀TensorRT-Alpha will be updated synchronously as soon as possible!

# install yolov8
conda create -n yolov8 python==3.8 -y # for Linux
# conda create -n yolov8 python=3.9 -y # for Windows10
conda activate yolov8
pip install ultralytics==8.0.5
pip install onnx==1.12.0

# download offical weights(".pt" file)
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x6.pt
```

export onnx:
```bash
# 640
yolo mode=export model=yolov8n.pt format=onnx dynamic=False opset=12    #simplify=True
tensorrt-inference only support static input. so set dynamic=False.

```

YoloV8在对应机器上使用TersorRT转换Onnx模型为TensorRT模型。
./TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8n.onnx  --saveEngine=yolov8n.trt  --buildOnly

## 4.Run tensorrt_inference
cd tensorrt_inference/bin
linkdome@linkdome-System-Product-Name:/data2/huang-work/tensorRT/tensorrt_inference/bin$ ./tensorrt_inference yolov8 ../configs/yolov8/config.yaml ../samples/detection_segmentation

The output in ../samples/detection_segmentation

config.yaml
modify onnx path

yolov8:
    onnx_file:     "../weights/yolov8n.onnx"
    engine_file:   "../weights/yolov8n.trt"
    
    
## 4. yolov8n-pose

# install yolov8
conda create -n yolov8 python==3.8 -y # for Linux
# conda create -n yolov8 python=3.9 -y # for Windows10
conda activate yolov8
pip install ultralytics==8.0.200
pip install onnx==1.12.0

# download offical weights(".pt" file)
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

export onnx:
yolo mode=export model=yolov8n-pose.pt format=onnx dynamic=False opset=12
./TensorRT-8.4.1.5/bin/trtexec   --onnx=yolov8n-pose.onnx  --saveEngine=yolov8n-pose.trt  --buildOnly

## add c files
./tensorrt_inference_yolov8pose/code/includes/yolov8Pose.h 
./tensorrt_inference_yolov8pose/code/src/yolov8Pose.cpp
./tensorrt_inference_yolov8pose/configs/yolov8Pose
./tensorrt_inference_yolov8pose/project/yolov8Pose
./tensorrt_inference_yolov8pose/project/main.cpp add yolov8Pose code.
./tensorrt_inference_yolov8pose/weights/yolov8n-pose.onnx


## make tensorrt_inference
cd tensorrt_inference/project
mkdir build && cd build
cmake ..
make -j

## .Run tensorrt_inference
cd tensorrt_inference/bin
linkdome@linkdome-System-Product-Name:/data2/huang-work/tensorRT/tensorrt_inference/bin$ ./tensorrt_inference yolov8Pose ../configs/yolov8Pose/config-pose.yaml ../samples/detection_pose



The output in ../samples/detection_segmentation

config.yaml
modify onnx path

yolov8:
    onnx_file:     "../weights/yolov8n.onnx"
    engine_file:   "../weights/yolov8n.trt"
    
    
 ## use libyolov8Pose.so to infer
./tensorrt_inference_yolov8pose/example-yolov8pose/README.md
 
    



















