from fastapi import FastAPI
from pydantic import BaseModel
from app import ask_pdf

app = FastAPI()

# Request format
class Query(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
def ask(query: Query):
    answer = ask_pdf(query.question)

    # Future HubSpot logic will go here
    if answer == "ESCALATE":
        return {
            "status": "escalate",
            "answer": "I couldn't find the answer. Our team will contact you."
        }

    return {
        "status": "success",
        "answer": answer
    }