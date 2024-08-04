import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from transformers import ResNetForImageClassification, AutoImageProcessor
import os
from fastapi import FastAPI

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

@app.get("/")
def home():
    # Retorna un simple mensaje de texto
    return 'Hola mundo: Model API - VERSION 1'
# Configuración del dispositivo
device = "cpu"


# Configurar el tamaño máximo de fragmento de memoria en MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:30'

# Cargar los modelos
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

@app.post("/classify_image")
def classify_image(image):
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
     logits = model(**inputs).logits

   
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]