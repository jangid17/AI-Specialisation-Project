"""Paths, model names and limits. Values come from the environment / .env (see env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("RA_DATA_DIR", BASE_DIR / "data"))
LIBRARY_DIR = DATA_DIR / "library"
INDEX_DIR = DATA_DIR / "index"
PROFILE_FILE = DATA_DIR / "profile.json"
MODEL_CACHE_DIR = DATA_DIR / "models"  # fastembed's default is the system temp folder, which reboots clear

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("RA_CHAT_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("RA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval settings, chosen by the ablation study in eval/ (see eval/README.md for the numbers).
# Changing CHUNKING or EMBEDDING_MODEL re-indexes the saved papers on the next search.
CHUNKING = os.getenv("RA_CHUNKING", "sentence-512")
RETRIEVAL_METHOD = os.getenv("RA_RETRIEVAL", "hybrid+rerank")
RERANK_MODEL = os.getenv("RA_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 30  # passages the cross-encoder re-scores per query
ANSWER_PASSAGES = 12     # passages (~5,300 characters) sent to the model for a question about one paper
LIBRARY_PASSAGES = 16    # ... and for a question about the whole library

# Guardrails (research_assistant/guardrails.py). Both models are on Groq's free tier with their own limits.
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"  # returns the probability of a prompt attack
SAFETY_MODEL = "openai/gpt-oss-safeguard-20b"               # classifies text against a written policy
INJECTION_THRESHOLD = 0.5
RELEVANCE_THRESHOLD = -5.0  # best re-ranker score below this: nothing relevant (calibrated in eval/results/final.json)
MAX_QUESTION_CHARS = 1000

# Groq's free tier allows 8,000 tokens per minute per model, counting prompt and completion together.
# One request therefore has to fit well inside that, and the context budgets below are sized for it.
MAX_OUTPUT_TOKENS = int(os.getenv("RA_MAX_OUTPUT_TOKENS", "1800"))
CONTEXT_CHAR_BUDGET = int(os.getenv("RA_CONTEXT_CHARS", "15000"))  # ~3,750 tokens of paper text

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL = 3.0  # seconds between API calls, per arXiv's terms of use
USER_AGENT = "ai-research-assistant/2.0 (academic reading tool; +https://github.com/sohamkkk/AI-Research-Assistant-New)"
HTTP_TIMEOUT = 30


def groq_api_key() -> str | None:
    """The key from the environment, or from Streamlit secrets when deployed."""
    for name in ("GROQ_API_KEY", "GROQ_CLOUD_API_KEY"):
        if os.getenv(name):
            return os.getenv(name)
    try:
        import streamlit as st

        for name in ("GROQ_API_KEY", "GROQ_CLOUD_API_KEY"):
            if name in st.secrets:
                return st.secrets[name]
    except Exception:  # no secrets file, or not running under Streamlit
        pass
    return None
