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
router = APIRouter()
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

OPENAI_API_KEY = "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"
system_template = """Use the following pieces of context to answer the users question.
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



    print("kooi")

    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=embeddings,
                                     persist_directory=DB_DIR)
    print("kooi2")
    vectordb.persist()

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a ChatOpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)
    print("kooi3")
    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Run the prompt and return the response
    response = qa(question)

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


def clean(transcript):
    from string import punctuation

    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_lg')

    doc = nlp(transcript)
    punctuated_text = ''
    for token in doc:
        # Add a space after each token
        punctuated_text += token.text + ' '
        # If the token ends a sentence, add a period
        if token.is_sent_end:
            punctuated_text = punctuated_text[:-1] + '. '

    # Print the punctuated text
    # print(punctuated_text)
    tokens = [token.text for token in doc]
    # print(tokens)
    punctuation = punctuation + '\n'
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    # print(word_frequencies)
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
        # print(word_frequencies)
    sentence_tokens = [sent for sent in doc.sents]
    # print(sentence_tokens)
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    return summary


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