
# Age_Agender => ONNX => TensorRT


## 1.Export ONNX Model

```bash
python convert_to_onnx.py --model_path /path/to/ResnetAgender.pth \ 
                          --image_path /path/to/input_image.jpg \ 
                          --onnx_save_path /path/to/save/ResnetAgender.onnx

```

## 2.Inference onnx
```bash
python infer_onnx.py --onnx_model_path /path/to/model.onnx --image_path /path/to/image.jpg

```

## 3.Build tensorrt_inference Project
```bash
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run tensorrt_inference
```bash
cd ../../bin/
./tensorrt_inference ResnetAgender ../configs/ResnetAgender/config.yaml ../samples/your_data
```

## 5.detect results