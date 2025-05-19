import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms as T
import argparse

######################################################################
# Settings
# ---------

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['baby', 'child', 'Teenager', 'Youth', 'Middle-age', 'old', 'gender']

######################################################################
# Argument Parsing
# ----------------

parser = argparse.ArgumentParser(description="Run inference on an ONNX model for age and gender prediction.")
parser.add_argument('--onnx_model_path', type=str, required=True, help="Path to the ONNX model file.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file.")
args = parser.parse_args()

######################################################################
# Load ONNX model
# ---------------

onnx_model_path = args.onnx_model_path
ort_session = ort.InferenceSession(onnx_model_path)

######################################################################
# Load Image and Preprocess
# -------------------------

def load_image(path):
    # Read and convert the image
    src1 = cv2.imread(path)
    example = Image.fromarray(cv2.cvtColor(src1, cv2.COLOR_BGR2RGB))
    example = transforms(example).unsqueeze(0).numpy()  # Convert to numpy for ONNX
    return example

image_path = args.image_path
input_image = load_image(image_path)

######################################################################
# ONNX Inference
# --------------

# ONNX expects input in the format (N, C, H, W) and numpy arrays.
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: input_image})

out1 = outputs[0]

######################################################################
# Postprocessing
# --------------

# Split into age and gender predictions
age = out1[:, 0:6]
gender = out1[:, 6:7]

result = {}

# Gender prediction
if float(gender) < 0.5:
    result['gender'] = 'Female'
    result['gender_confident'] = 1.0 - float(gender)
else:
    result['gender'] = 'Male'
    result['gender_confident'] = float(gender)

# Age prediction
age1 = age[0]  # Only one image is processed, so take the first batch
a_index = np.argmax(age1)
a_max = np.max(age1)

if 0 <= a_index <= 1:
    age_is = "0-20"
elif 2 <= a_index <= 3:
    age_is = '21-40'
elif a_index == 4:
    age_is = '41-60'
elif a_index == 5:
    age_is = '60+'
result['age'] = age_is
result['age_confident'] = float(a_max)

print(f"raw out: {out1}\n{image_path}")
print(result)
