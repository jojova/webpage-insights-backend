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
from urllib.parse import urlparse, urljoin
from html.parser import HTMLParser
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)


def scrape_relevant_paragraphs(url, min_paragraph_length=50, keywords=None):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all paragraph elements
        paragraphs = soup.find_all('p')

        # Extract text from paragraphs and filter based on length and keywords
        relevant_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) >= min_paragraph_length:
                if keywords:
                    if any(keyword.lower() in text.lower() for keyword in keywords):
                        relevant_paragraphs.append(text)
                else:
                    relevant_paragraphs.append(text)

        # Return relevant text paragraphs
        return relevant_paragraphs
    else:
        print(f"Failed to fetch URL: {response.status_code}")
        return []
class ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.image_urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            for attr in attrs:
                if attr[0] == 'src':
                    self.image_urls.append(attr[1])

def scrape_images(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        parser = ImageParser()
        parser.feed(response.text)

        # Convert relative URLs to absolute URLs
        base_url = urlparse(url)
        absolute_image_urls = [urljoin(base_url.scheme + '://' + base_url.netloc, img_url) for img_url in parser.image_urls]

        return absolute_image_urls
    else:
        print(f"Failed to fetch URL: {response.status_code}")
        return []
    
@router.post("/text/")
async def scrape_content(url:str):
    try:
        return {"response": scrape_relevant_paragraphs(url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/")
async def scrape_content(url:str):
    try:
        return {"response": scrape_images(url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

