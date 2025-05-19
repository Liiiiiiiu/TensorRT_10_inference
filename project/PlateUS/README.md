
# PlateUS => ONNX => TensorRT


## 1.Export ONNX Model

```bash
python export.py 

```

## 2.Inference onnx
```bash
python infer_onnx.py 

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
./tensorrt_inference PlateUS ../configs/PlateUS/config.yaml ../samples/plate
```

## 5.detect results
