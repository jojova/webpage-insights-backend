from io import StringIO

from fastapi import HTTPException, UploadFile, File, APIRouter
from dotenv import load_dotenv
from pandasai.responses.response_serializer import ResponseSerializer

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

load_dotenv()

OPENAI_API_KEY = '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'
llm = OpenAI(api_token=OPENAI_API_KEY)
router = APIRouter()


def get_response(file:StringIO, question:str):
	df = pd.read_csv(file)
	chat_df = SmartDataframe(df, config={"llm": llm})
	response = chat_df.chat(question)
	if isinstance(response,str):
		return response
	raise HTTPException(status_code=500, detail="only chat is supported")

def get_piechart(file:StringIO, question:str):
	df = pd.read_csv(file)
	chat_df = SmartDataframe(df, config={"llm": llm, "save_charts": True,})
	response = chat_df.chat(question)
	return response


@router.post("/query/")
async def query_website(query: str, file: UploadFile = File(...), ):
	try:
		if file.filename.endswith('.csv'):
			# Read the contents of the uploaded file
			contents = await file.read()
			# Convert to a StringIO object to mimic a file object for pandas
			data = StringIO(contents.decode("utf-8"))
			return {"response": get_response(data, query)}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@router.post("/chart/")
async def query_website_chart(query: str, file: UploadFile = File(...), ):
	try:
		if file.filename.endswith('.csv'):
			contents = await file.read()
			data = StringIO(contents.decode("utf-8"))
			return {"response": get_piechart(data, query)}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


