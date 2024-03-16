"""Python FastAPI Auth0 integration example
"""
from openai import OpenAI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Security
from utils import VerifyToken

# Creates app instance
app = FastAPI()
auth = VerifyToken()


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



@app.get("/api/private-scoped")
def private_scoped(auth_result: str = Security(auth.verify, scopes=['read:messages'])):
    """A valid access token and an appropriate scope are required to access
    this route
    """

    return auth_result


