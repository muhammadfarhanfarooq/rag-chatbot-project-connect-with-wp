from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app import ask_pdf

app = FastAPI()

# Allow WordPress site to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We will restrict this to your WordPress URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class Query(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
def ask(query: Query):
    answer = ask_pdf(query.question)

    if answer == "ESCALATE":
        return {
            "status": "escalate",
            "answer": "I couldn't find the answer. Our team will contact you."
        }

    return {
        "status": "success",
        "answer": answer
    }
