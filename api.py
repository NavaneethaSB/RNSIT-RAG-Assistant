"""FastAPI interface for the existing hybrid RNSIT RAG pipeline."""

from contextlib import asynccontextmanager
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ask import generate_answer_with_source, load_resources, retrieve_chunks, select_relevant_chunks

_FOLLOWUP_FILLER_WORDS = {
    "a", "an", "and", "are", "at", "about", "can", "could", "describe", "did",
    "do", "does", "for", "from", "give", "how", "in", "is", "it", "its", "me",
    "more", "of", "on", "or", "please", "should", "tell", "that", "the", "their",
    "them", "these", "they", "this", "those", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "would", "explain",
}


def _retrieval_terms(text: str) -> list[str]:
    """Keep topic and request terms while dropping question-form filler."""
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _FOLLOWUP_FILLER_WORDS
    ]


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    conversation: list[ConversationTurn] = Field(default_factory=list, max_length=8)


class Source(BaseModel):
    id: str
    title: str
    url: str
    excerpt: str
    retrieval_rank: int
    retrieval_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resource/model loading is intentionally performed once, not per question.
    app.state.rag_resources = await run_in_threadpool(load_resources)
    yield


app = FastAPI(title="RNSIT AI Assistant API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Report readiness only after the retrieval pipeline loaded successfully."""
    resources = getattr(request.app.state, "rag_resources", None)
    if not isinstance(resources, tuple) or len(resources) != 5 or any(item is None for item in resources):
        raise HTTPException(status_code=503, detail="The RNSIT knowledge service is not ready.")
    return {"status": "ok"}


def _retrieval_query(question: str, conversation: list[ConversationTurn]) -> str:
    """Build a focused query from the prior user topic and follow-up terms."""
    if not conversation or not re.search(
        r"\b(more|it|its|that|those|them|their|there|these|also)\b|^and\b",
        question.strip(), re.IGNORECASE,
    ):
        return question
    prior_questions = [turn.content.strip() for turn in conversation[-6:] if turn.role == "user"][-1:]
    if not prior_questions:
        return question
    prior_terms = _retrieval_terms(prior_questions[0][:500])
    current_terms = _retrieval_terms(question)
    # Avoid repeating interrogative scaffolding from the old question. Those
    # common terms can dominate BM25 and push the actual topic out of its pool.
    query_terms = list(dict.fromkeys(prior_terms + current_terms))
    return " ".join(query_terms) if query_terms else question


def _needs_clarification(question: str, conversation: list[ConversationTurn]) -> bool:
    has_previous_question = any(turn.role == "user" for turn in conversation)
    generic_followup = re.fullmatch(
        r"(?:tell me more|more|what about it|how about that|what is its(?: duration)?|how long is it)[?.! ]*",
        question.strip(),
        re.IGNORECASE,
    )
    return bool(generic_followup and not has_previous_question)


@app.post("/api/chat", response_model=AskResponse)
async def chat(payload: AskRequest, request: Request) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a question.")
    if _needs_clarification(question, payload.conversation):
        return AskResponse(
            answer="What would you like me to expand on? Please mention a department or programme.",
            sources=[],
        )

    chunks, index, embedding_model, bm25, client = request.app.state.rag_resources
    retrieval_query = _retrieval_query(question, payload.conversation)

    def answer_question() -> tuple[str, list[dict[str, object]]]:
        retrieved = retrieve_chunks(
            retrieval_query,
            chunks,
            index,
            embedding_model,
            bm25,
        )

        retrieved = select_relevant_chunks(
            retrieval_query,
            retrieved,
            chunks,
        )

        answer, source = generate_answer_with_source(
            question,
            retrieved,
            client,
            chunks,
        )

        if source is None:
            sources = []
        elif isinstance(source, list):
            sources = source
        else:
            sources = [source]
        return answer, sources

    try:
        answer, sources = await run_in_threadpool(answer_question)
    except Exception as error:
        # Groq's client enforces a 45 second upstream timeout. Surface a clear
        # gateway response without returning internal details or credentials.
        error_name = type(error).__name__.lower()
        if "timeout" in error_name:
            raise HTTPException(status_code=504, detail="The answer service timed out. Please try again.") from error
        raise HTTPException(status_code=502, detail="The answer service could not complete your request. Please try again.") from error

    return AskResponse(answer=answer, sources=sources)
