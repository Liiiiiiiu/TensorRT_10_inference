import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import onnxruntime as ort
from resnet18 import resnet18

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('=> device: ', device)

class Classifier(torch.nn.Module):
    def __init__(self, num_cls, input_size, is_freeze=False):
        super(Classifier, self).__init__()
        self.features = resnet18(num_cls)

        if is_freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, X):
        return self.features(X)

def load_model(model_path, num_classes, input_size, device):
    model = Classifier(num_classes, input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return image, preprocess(image).unsqueeze(0).numpy()

def predict_onnx(onnx_model_path, image, color_attrs):
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    raw_outputs = ort_session.run(None, ort_inputs)[0]
    
    probabilities = softmax(raw_outputs, axis=1)
    predicted_label = np.argmax(probabilities)
    return predicted_label, color_attrs[predicted_label], probabilities[0][predicted_label]

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def process_images_in_folder(input_folder, output_folder, onnx_model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    color_attrs = ["Black", "Blue", "Gray", "Green", "Pink", "Red", "White", "Yellow"]
    input_size = 224  # ResNet18 expected input size

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            original_image, image_tensor = preprocess_image(image_path, input_size)
            label, color, confidence = predict_onnx(onnx_model_path, image_tensor, color_attrs)
            
            # Draw the prediction on the image
            draw = ImageDraw.Draw(original_image)
            font = ImageFont.load_default()
            text = f"Predicted: {color}, Confidence: {confidence:.2f}"
            draw.text((0, 0), text, font=font, fill="red")
            
            # Save the image with the prediction
            output_image_path = os.path.join(output_folder, image_name)
            original_image.save(output_image_path)
            print(f"Processed {image_name} -> {color} with confidence {confidence:.2f}")

def main(input_folder, output_folder, export_onnx):
    num_classes = 8  # Number of color classes
    input_size = 224  # Model input size

    if export_onnx:
        model = load_model('path/to/you.pth', num_classes, input_size, device)
        export_model_to_onnx(model, os.path.join(output_folder, 'xxx.onnx'), input_size)
    else:
        onnx_model_path = 'path/to/you.onnx'
        process_images_in_folder(input_folder, output_folder, onnx_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Color Classification")
    parser.add_argument("--input_folder", type=str, default='path/to/you')
    parser.add_argument("--output_folder", type=str, default='path/to/you')
    parser.add_argument("--export_onnx", action='store_true')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.export_onnx)
