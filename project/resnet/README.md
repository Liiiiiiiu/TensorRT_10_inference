# ResNet => ONNX => TensorRT

## 1. Export and Inference with ONNX Model

Before running the commands below, ensure the correct paths for both `.pth` and `.onnx` files are configured in the script.

### 1.1 Vehicle Color Classification: Export to ONNX and Inference
To export the vehicle color classification model to ONNX format and perform inference, run the following command:
```bash
python export_infer_vehicle_color.py --input_folder /path/to/your/input \
                                     --output_folder /path/to/your/output \
                                     --export_onnx
```

### 1.2 Vehicle Type Classification: Export to ONNX and Inference
To export the vehicle type classification model to ONNX format and perform inference, run the following command:
```bash
python export_infer_vehicle_type.py --input_folder /path/to/your/input \
                                     --output_folder /path/to/your/output \
                                     --export_onnx
```

## 2.Build tensorrt_inference Project
```bash
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 3.run tensorrt_inference
```bash
cd ../../bin/
./tensorrt_inference resnet ../configs/resnet/config.yaml ../samples/your_data
```

## 4.detect results