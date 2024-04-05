from fastapi import FastAPI
from server.authentication import router as auth_router
from webchat.view import router as webchat_router
from summarize.view import router as summarize_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="WPI API",
    version="0.5.0",
)

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, tags=["Auth"],)
app.include_router(webchat_router, tags=["Webchat"], prefix="/webchat")
app.include_router(summarize_router, tags=["Summarize"], prefix="/summarize")

@app.get("/", tags=["Test"])
async def test_response():
    return "Api Start"


app.mount("/exports", StaticFiles(directory="exports"), name="exports")
