"""FastAPI backend for the Interactive Campus Info AI Agent.

This is a minimal, production-ready-ish starter that:
- Exposes a POST /ask endpoint for chatbot queries
- Proxies questions to a Groq-hosted LLM
- Returns the model's response back to the frontend

Scraping and vector search (RAG) are intentionally omitted at this stage.
"""

from typing import Any, Dict

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    # Fail fast if API key is missing so misconfiguration is obvious.
    raise RuntimeError("GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=<your_key>.")


class AskRequest(BaseModel):
    """Request body schema for /ask endpoint.

    Attributes
    ----------
    question: str
        The student's question or message to the chatbot.
    """

    question: str


class AskResponse(BaseModel):
    """Response body schema for /ask endpoint."""

    answer: str


app = FastAPI(title="Interactive Campus Info AI Agent", version="0.1.0")

# Configure CORS to allow the frontend (served from the same origin or file://) to call the API.
# In production, you should restrict allowed_origins to your real frontend domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def call_groq_llm(question: str) -> str:
    """Call Groq's Chat Completions API with the user's question.

    Parameters
    ----------
    question: str
        The user's natural-language question.

    Returns
    -------
    str
        The assistant's answer text.
    """

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping students with information about "
                    "Kakatiya University College of Engineering and Technology. "
                    "Answer clearly and concisely. If you are unsure, say you are not sure."
                ),
            },
            {"role": "user", "content": question},
        ],
        # Keep it simple and cheap for now; tune as needed.
        "temperature": 0.2,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        except httpx.RequestError as exc:
            # Network-level issues
            raise HTTPException(status_code=502, detail=f"Error calling Groq API: {exc}") from exc

    if response.status_code != 200:
        # Surface detailed error information for easier debugging (but avoid leaking secrets).
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Groq API returned an error",
                "status_code": response.status_code,
                "body": response.text,
            },
        )

    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected response format from Groq API") from exc

    return answer


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Chatbot endpoint.

    Accepts a student's question, forwards it to the LLM, and returns the answer.
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = await call_groq_llm(request.question)
    return AskResponse(answer=answer)


@app.get("/")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint to verify the API is running."""

    return {"status": "ok"}
