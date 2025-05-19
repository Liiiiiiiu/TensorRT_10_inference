cmake_minimum_required(VERSION 3.5)
project(tensorrt_inference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug) # Enable debug mode

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_DIR /mnt/jx/tensorrt_inference_yolov8pose/TensorRT-10.3.0.26)
set(TENSORRT_LIB_DIR
    ${TENSORRT_DIR}/targets/x86_64-linux-gnu/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
)
set(TENSORRT_INCLUDE_DIR
    ${TENSORRT_DIR}/targets/x86_64-linux-gnu/include
    /usr/local/include
    /usr/include/x86_64-linux-gnu
)

find_library(TENSORRT_LIBRARY_INFER NAMES nvinfer libnvinfer.so HINTS ${TENSORRT_LIB_DIR})
find_library(TENSORRT_LIBRARY_ONNXPARSER NAMES nvonnxparser libnvonnxparser.so HINTS ${TENSORRT_LIB_DIR})
find_library(TENSORRT_LIBRARY_PLUGIN NAMES nvinfer_plugin libnvinfer_plugin.so HINTS ${TENSORRT_LIB_DIR})

if(NOT TENSORRT_LIBRARY_INFER)
    message(FATAL_ERROR "Cannot find libnvinfer.so in ${TENSORRT_LIB_DIR}")
endif()
if(NOT TENSORRT_LIBRARY_ONNXPARSER)
    message(FATAL_ERROR "Cannot find libnvonnxparser.so in ${TENSORRT_LIB_DIR}")
endif()
if(NOT TENSORRT_LIBRARY_PLUGIN)
    message(FATAL_ERROR "Cannot find libnvinfer_plugin.so in ${TENSORRT_LIB_DIR}")
endif()

set(TENSORRT_LIBRARIES
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_ONNXPARSER}
    ${TENSORRT_LIBRARY_PLUGIN}
)
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libraries: ${TENSORRT_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")

# YAML-CPP
set(YAML_INCLUDE /mnt/jx/tensorrt_inference_yolov8pose/depends/yaml-cpp-master/include)
set(YAML_LIB_DIR /mnt/jx/tensorrt_inference_yolov8pose/depends/yaml-cpp-master/build)
find_library(YAML_LIBRARY NAMES yaml-cpp libyaml-cpp.so libyaml-cpp.so.0.8 HINTS ${YAML_LIB_DIR})
if(NOT YAML_LIBRARY)
    message(FATAL_ERROR "yaml-cpp library not found in ${YAML_LIB_DIR}")
endif()
message(STATUS "YAML include: ${YAML_INCLUDE}")
message(STATUS "YAML library: ${YAML_LIBRARY}")

# Model includes
set(MODEL_INCLUDE ../code/includes)

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${MODEL_INCLUDE}
    ${YAML_INCLUDE}
)

# Link directories
link_directories(
    ${TENSORRT_LIB_DIR}
    ${YAML_LIB_DIR}
)

# Source files
set(SOURCES
    main.cpp
    ../code/src/build.cpp
)

# Subdirectories
add_subdirectory(resnet)
add_subdirectory(yolov5_cu)
add_subdirectory(yolov8)
add_subdirectory(fast-reid)
add_subdirectory(CenterFace)

# Factory library
add_library(factory SHARED ../code/src/build.cpp)

# Executable
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin)
add_executable(tensorrt_inference ${SOURCES})
target_link_libraries(tensorrt_inference
    factory
    resnet
    yolov5_cu
    yolov8
    fast-reid
    CenterFace
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${YAML_LIBRARY}
    pthread
)