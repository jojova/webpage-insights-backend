from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
import secrets
from dotenv import load_dotenv
from .models import UserModel, UserModelOut, Token, TokenData
from passlib.context import CryptContext
import re
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from .database import database
import os


router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login/")
load_dotenv()

MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
ALGORITHM = os.environ.get("ALGORITHM")
SECRET_KEY=os.environ.get("SECRET_KEY")
TOKEN_EXPIRES = int(os.environ.get("TOKEN_EXPIRES"))
USER_COLLECTION = os.environ.get("USER_COLLECTION")



user_collection = database[USER_COLLECTION]


def decode_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        elif not user_collection.find_one({"email_id": username}):
            raise HTTPException(status_code=401, detail="Invalid token")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if user_collection.find_one({"email_id": username}):
        user_dict = user_collection.find_one({"email_id": username})
        return UserModel(**user_dict)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(fake_db, username: str, password: str,):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(MONGO_DB_NAME, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserModelOut = Depends(get_current_user)):
    return current_user


@router.post("/login/", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(os.environ.get(
        "MONGO_DB_NAME"), form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=120)
    access_token = create_access_token(
        data={"sub": user.email_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/current_user/", response_model=UserModelOut)
async def read_users_me(current_user: UserModelOut = Depends(get_current_active_user)):
    return current_user


@router.post("/register/", description='Enter User Detail and Call on Form Submit', status_code=status.HTTP_201_CREATED)
async def create_list(lists: UserModel = Body(...)):
    email_pattern = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    if user_collection.find_one({"email_id": lists.email_id}):
        raise HTTPException(status_code=400, detail="User already exists", headers={
                            "X-Error": "Duplicate"})
    if not email_pattern.match(lists.email_id):
        raise HTTPException(status_code=400, detail="Invalid email address")
    if lists.password=="":
        raise HTTPException(status_code=400, detail="Password is required",header={
             "X-Error": "Empty Password"})
    lists.password = get_password_hash(lists.password)
    lists_dict = lists.dict()
    new_list_item = user_collection.insert_one(lists_dict)
   
    created_list_item = user_collection.find_one({
        "_id": new_list_item.inserted_id
    })

    return {"msg": "User successfully Created", "data": UserModelOut(**created_list_item)}



