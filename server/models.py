from typing import Optional
from bson import ObjectId
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr, Field
from enum import Enum
import uuid


def generate_uuid():
    return str(uuid.uuid4()[:6]).replace("-", "")


class UserModelOut(BaseModel):
    fname: str = Field(...)
    lname: str = Field(...)
    email_id: str = Field(default=...)

    class Config:
        schema_extra = {
            "example": {
                "fname": "John",
                "lname": "Thrikkakara",
                "email_id": 'jdoe@hotmail.com',
            }
        }


class UserModel(UserModelOut):
    fname: str = Field(...)
    lname: str = Field(...)
    email_id: str = Field(default=...)
    password: str = Field(...)
    deleted: int = Field(default=0)

    class Config:
        schema_extra = {
            "example": {
                "fname": "John",
                "lname": "Thrikkakara",
                "email_id": 'jdoe@hotmail.com',
                "password": "passpass",
                "deleted": 0
            }
        }


class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        schema_extra = {
            "example": {
                "accestoken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI5NGQ1OWNlZi1kYmI4LTRlYTUtYjE3OC1kMjU0MGZjZDY5MTkiLCJqdGkiOiI2Yj",
                "token_type": "Bearer",
            }
        }


class TokenData(BaseModel):
    username: str | None = None


def ResponseModel(data, message):
    return {
        "data": [data],
        "code": 200,
        "message": message,
    }


def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}
