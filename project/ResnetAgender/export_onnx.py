import torch
from PIL import Image
from torchvision import transforms as T
from net import get_model
import onnx
from onnxsim import simplify
import argparse

# Define transformations for input image
transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and transform image for model input
def load_image(path):
    src = Image.open(path)
    src = transforms(src).unsqueeze(dim=0).cuda()  # Add batch dimension and move to GPU
    return src

# Load model and its weights
def load_network(model_path):
    model_name = 'resnet50_nfc'
    model = get_model(model_name, num_label=7, use_id=False, num_id=1000)
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.cuda()
    model.eval()  # Set model to evaluation mode
    print(f'Model loaded from {model_path}')
    return model

# Convert model to ONNX format with optional simplification
def convert_onnx_model(model, save_path, img, simp=False):
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}  # Set dynamic axes for batch size

    # Export the model to ONNX format
    torch.onnx.export(model, img, save_path, input_names=input_names, output_names=output_names,
                      opset_version=12, dynamic_axes=dynamic_axes)
    
    # Simplify the ONNX model if specified
    if simp:
        onnx_model = onnx.load(save_path)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path)
        print(f'ONNX model simplified and saved at {save_path}')
    else:
        print(f'ONNX model saved at {save_path}')

# Main script
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to ONNX format")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the PyTorch model (.pth) file")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image for testing")
    parser.add_argument('--onnx_save_path', type=str, required=True, help="Path to save the ONNX model")
    args = parser.parse_args()

    # Load model and image, then perform ONNX conversion
    model = load_network(args.model_path)
    img = load_image(args.image_path)
    convert_onnx_model(model, args.onnx_save_path, img, simp=True)
