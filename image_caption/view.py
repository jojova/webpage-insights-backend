import re
import openai
import spacy
from fastapi import HTTPException, APIRouter
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
import urllib.parse
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.document_loaders import WebBaseLoader
router = APIRouter()
import os
from bs4 import BeautifulSoup
from string import punctuation
from typing import List
import requests 
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from io import BytesIO
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

model = VisionEncoderDecoderModel.from_pretrained("vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def open_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4XX and 5XX status codes
        
        # Create a temporary file to save the image
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # Open the image using PIL
        image = Image.open(temp_file_path)
        
        # Return the PIL image object and the path to the temporary file
        return image, temp_file_path
    except Exception as e:
        print(f"Error opening image from URL: {url}")
        print(e)
        return None, None
    
def predict_step(image_paths):
    images = []
    temp_file_paths = []

    for image_path in image_paths:
        if image_path.startswith("http"):
            # If the image path is a URL
            image, temp_file_path = open_image_from_url(image_path)
            temp_file_paths.append(temp_file_path)
            if image is None:
                continue  # Skip to the next image path
        else:
            # If the image path is a local file path
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images.append(image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    # Delete temporary files if any
    for temp_file_path in temp_file_paths:
        if temp_file_path:
            os.unlink(temp_file_path)
    
    return preds

@router.post("/image/")
async def scrape_content(url:List[str]):
    try:
        return {"response": predict_step(url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

