"""The reader's library: one JSON file per paper under data/library, plus a small profile file."""

import json
from datetime import datetime
from pathlib import Path

from research_assistant import config
from research_assistant.models import Paper


def _safe(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def paper_path(arxiv_id: str) -> Path:
    return config.LIBRARY_DIR / f"{_safe(arxiv_id)}.json"


def index_path(arxiv_id: str) -> Path:
    return config.INDEX_DIR / f"{_safe(arxiv_id)}.npz"


def save(paper: Paper) -> None:
    config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if not paper.added_at:
        paper.added_at = datetime.now().isoformat(timespec="seconds")
    tmp = paper_path(paper.arxiv_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(paper.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(paper_path(paper.arxiv_id))


def load(arxiv_id: str) -> Paper | None:
    path = paper_path(arxiv_id)
    if not path.exists():
        return None
    return Paper.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_all() -> list[Paper]:
    """Every paper in the library, most recently added first."""
    if not config.LIBRARY_DIR.exists():
        return []
    papers = [Paper.from_dict(json.loads(p.read_text(encoding="utf-8")))
              for p in config.LIBRARY_DIR.glob("*.json")]
    return sorted(papers, key=lambda p: p.added_at, reverse=True)


def contains(arxiv_id: str) -> bool:
    return paper_path(arxiv_id).exists()


def delete(arxiv_id: str) -> None:
    paper_path(arxiv_id).unlink(missing_ok=True)
    index_path(arxiv_id).unlink(missing_ok=True)


DEFAULT_PROFILE = {"interests": "", "watchlist": []}


def load_profile() -> dict:
    """The reader's interests statement and saved searches: {"interests": str, "watchlist": [...]}."""
    if config.PROFILE_FILE.exists():
        return {**DEFAULT_PROFILE, **json.loads(config.PROFILE_FILE.read_text(encoding="utf-8"))}
    return dict(DEFAULT_PROFILE)


def save_profile(profile: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROFILE_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
