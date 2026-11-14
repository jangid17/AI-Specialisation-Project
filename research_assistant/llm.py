"""Chat completions from Groq through its OpenAI-compatible endpoint."""

import json
import re
import time
from collections.abc import Callable, Iterator

import openai

from research_assistant import config

MAX_RETRIES = 4
MAX_WAIT = 65.0  # the free-tier token window is one minute

_client: openai.OpenAI | None = None


class LLMError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(config.groq_api_key())


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        key = config.groq_api_key()
        if not key:
            raise LLMError("No Groq API key found. Put GROQ_API_KEY=... in the .env file "
                           "(free keys: https://console.groq.com/keys).")
        # Retries are handled here so the caller can tell the reader how long the wait is.
        _client = openai.OpenAI(api_key=key, base_url=config.GROQ_BASE_URL, max_retries=0, timeout=120)
    return _client


def _retry_after(err: openai.APIStatusError) -> float:
    header = err.response.headers.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, MAX_WAIT)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)(ms|s)", str(err))
    if m:
        seconds = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        return min(seconds + 0.5, MAX_WAIT)
    return 20.0


def _create(messages: list[dict], max_tokens: int, json_mode: bool, stream: bool,
            on_wait: Callable[[float], None] | None, model: str | None = None):
    model = model or config.CHAT_MODEL
    kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                    "temperature": 0.2, "stream": stream}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"  # reasoning tokens count against the per-minute budget
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client().chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise LLMError("Groq's rate limit is still exceeded after several waits. "
                               "Try again in a minute.") from e
            wait = _retry_after(e)
            if on_wait:
                on_wait(wait)
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code == 413:
                raise LLMError("The request is larger than Groq's per-minute token allowance. "
                               "Lower RA_CONTEXT_CHARS or use a shorter selection.") from e
            raise LLMError(f"Groq returned an error ({e.status_code}): {e.message}") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Could not reach Groq: {e}") from e


def chat(messages: list[dict], max_tokens: int = config.MAX_OUTPUT_TOKENS, json_mode: bool = False,
         on_wait: Callable[[float], None] | None = None, model: str | None = None) -> str:
    """One completion, from the chat model unless `model` names another (e.g. a guard model).

    `on_wait(seconds)` is called before each rate-limit pause.
    """
    resp = _create(messages, max_tokens, json_mode, False, on_wait, model)
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise LLMError("The model returned an empty answer (it may have run out of output tokens).")
    return content


def stream(messages: list[dict], max_tokens: int = config.MAX_OUTPUT_TOKENS,
           on_wait: Callable[[float], None] | None = None) -> Iterator[str]:
    """Yield the answer text as it is generated."""
    for chunk in _create(messages, max_tokens, False, True, on_wait):
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("The model did not return valid JSON.") from None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError("The model did not return valid JSON.") from e


def chat_json(messages: list[dict], max_tokens: int = config.MAX_OUTPUT_TOKENS,
              on_wait: Callable[[float], None] | None = None) -> dict:
    return parse_json(chat(messages, max_tokens, json_mode=True, on_wait=on_wait))
