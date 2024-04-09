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
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

OPENAI_API_KEY = "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"
system_template = """Use the following pieces of context only to answer the users question. Don't use other knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def generate_embeddings(text):
    client = OpenAI()
    response = openai.Embedding.create(
        model="text-embedding-3-large",  # Specify the embedding model you want to use
        input=text  # The text for which to generate embeddings
    )
    embeddings = response['data'][0]['embedding']  # Extracting the embedding
    return embeddings


def get_response(paragraph, question):
    # Assuming the directory setup for database persistence
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # gneerate embeddings for the paragraph
     # Split the loaded data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(paragraph)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(pages)


    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=embeddings,
                                     persist_directory=DB_DIR)
    vectordb.persist()

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a ChatOpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)
    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Run the prompt and return the response
    response = qa("""Use the following pieces of context only to answer the users question. Don't use other knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer."""+question)

    return response


def get_transcript(url):
    # Parse the query string parameters of the URL
    query_params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)

    id = query_params['v'][0]
    ##id = 'Y8Tko2YC5hA'
    transcript = YouTubeTranscriptApi.get_transcript(id)
    script = ""

    for text in transcript:
        t = text["text"]
        if t != '[Music]':
            script += t + " "
    transcript, no_of_words = script, len(script.split())

    return transcript, no_of_words

def clean(transcript, include_first_sents=1, summary_proportion=0.3):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(transcript)

    sentence_scores = {}
    all_sentences = list(doc.sents)
    
    # Calculate word frequencies with a focus solely on the content
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.text.lower() not in word_frequencies:
                word_frequencies[word.text.lower()] = 1
            else:
                word_frequencies[word.text.lower()] += 1

    max_frequency = max(word_frequencies.values())
    
    # Score sentences based on word frequencies, ignoring sentence position and length
    for sent in all_sentences:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent in sentence_scores:
                    sentence_scores[sent] += word_frequencies[word.text.lower()] / max_frequency
                else:
                    sentence_scores[sent] = word_frequencies[word.text.lower()] / max_frequency

    # Normalize sentence scores
    max_score = max(sentence_scores.values()) if sentence_scores else 1
    for sent in sentence_scores:
        sentence_scores[sent] = sentence_scores[sent] / max_score

    # Ensure first sentences are included in the summary
    summary_sentences = all_sentences[:include_first_sents]
    select_length = max(int(len(all_sentences) * summary_proportion) - include_first_sents, 1)
    
    # Select top-scoring sentences for the remainder of the summary
    remaining_sentences = [s for s in all_sentences if s not in summary_sentences]
    remaining_scores = {s: score for s, score in sentence_scores.items() if s in remaining_sentences}
    top_sentences = nlargest(select_length, remaining_scores, key=remaining_scores.get)

    summary_sentences.extend(top_sentences)
    # Sort selected sentences into their original order
    summary_sentences = sorted(summary_sentences, key=lambda x: all_sentences.index(x))

    final_summary = ' '.join([sent.text for sent in summary_sentences])

    return final_summary

@router.post("/text/")
async def summarise(text: str):
    try:
        return {"response": clean(text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/youtube/text/")
async def summarise_youtube(url: str):
    try:
        transcript = get_transcript(url)
        return {"response": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/")
async def query_paragraph(query: str, paragraph: str):
    try:
        return {"response": get_response(paragraph, query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




