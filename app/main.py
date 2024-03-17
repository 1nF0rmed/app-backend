import os
import uuid
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Security
from prompts import generate_products_from_image
from utils import VerifyToken
from dotenv import load_dotenv
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")

# Creates app instance
app = FastAPI()
auth = VerifyToken()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/api/public")
def public():
    """No access token required to access this route"""

    result = {
        "status": "success",
        "msg": ("Hello from a public endpoint! You don't need to be "
                "authenticated to see this.")
    }
    return result


@app.get("/api/private")
def private(auth_result: str = Security(auth.verify)):
    """A valid access token is required to access this route"""
    return auth_result


class Prompt(BaseModel):
    content: str

@app.post("/api/gpt-response")
async def get_gpt_response(prompt: Prompt):
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt.content}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    user_id = str(uuid.uuid4())
    try:
        file_contents = await file.read()
        base_upload_folder = "uploads"
        user_upload_folder = f"{base_upload_folder}/{user_id}"
        if not os.path.exists(base_upload_folder):
            os.makedirs(base_upload_folder)
        if not os.path.exists(user_upload_folder):
            os.makedirs(user_upload_folder)
        file_name = str(uuid.uuid4())
        file_path = f"{user_upload_folder}/{file_name}"
        with open(file_path, "wb") as f:
            f.write(file_contents)
        return JSONResponse(status_code=200, content={"message": "File uploaded successfully", "data": file_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

class FilePrompt(BaseModel):
    data: str

@app.post("/api/analyze")
async def analyze(file_prompt: FilePrompt):
    products = generate_products_from_image(file_prompt.data)

    return products


@app.get("/api/db-connect")
def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")

        return {"ok": "success"}
    except Exception as e:
        print(e)


@app.get("/api/private-scoped")
def private_scoped(auth_result: str = Security(auth.verify, scopes=['read:messages'])):
    """A valid access token and an appropriate scope are required to access
    this route
    """

    return auth_result


