from PIL import Image
import torch
import torchvision.transforms as T
import gradio as gr
from torchvision import models
from torch import nn
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()  # Set model to evaluation mode

# Preprocess input for ResNet (same as what ResNet expects)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the captioning model (e.g., BLIP for simplicity)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_features(img):
    img_input = transform(img).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():  # Disable gradient calculation for feature extraction
        features = resnet_model(img_input)  # Extract features using ResNet
    return features

def generate_caption(img):
    # Extract features from image using ResNet
    features = extract_features(img)

    # Use BLIP model to generate caption
    inputs = processor(images=img, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Image"),
    outputs=gr.Text(label="Caption"),
)

demo.launch()
