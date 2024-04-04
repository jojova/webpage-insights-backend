from fastapi import FastAPI
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

mongodb_client = MongoClient(os.environ.get("MONGO_URL"))


def startup_db_client():
    database = mongodb_client[os.environ.get("MONGO_DB_NAME")]
    print("DB Connection Established")
    return database


database = startup_db_client()


def shutdown_db_client():
    mongodb_client.close()
    print("DB Connection Closed")
