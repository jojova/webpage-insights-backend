from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv("")

router = APIRouter()
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

OPENAI_API_KEY = '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'

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


class Query(BaseModel):
    url: str
    question: str

def get_response(url,question):
    # Load data from the specified URL
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split the loaded data
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=800,
                                          chunk_overlap=40)

    docs = text_splitter.split_documents(data)

    # Create OpenAI embeddings
    openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create a Chroma vector database from the documents
    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=openai_embeddings,
                                     persist_directory=DB_DIR)

    vectordb.persist()

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a ChatOpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',openai_api_key=OPENAI_API_KEY)

    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Run the prompt and return the response
    response = qa(question)

    print(response)


@router.post("/query/")
async def query_website(query: Query):
    try:
        return {"response":  "get_response(query.url, query.question)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
