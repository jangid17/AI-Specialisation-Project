"""The Paper record stored in the library."""

from dataclasses import asdict, dataclass, field
from datetime import date

STATUSES = ["To read", "Reading", "Read"]


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Paper:
    arxiv_id: str  # without version, e.g. "1706.03762"
    title: str
    authors: list[str]
    abstract: str
    published: str  # ISO date of the first version
    updated: str  # ISO date of the latest version
    primary_category: str
    categories: list[str] = field(default_factory=list)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    version: str = ""

    # Full text, filled by fulltext.fetch_full_text
    sections: list[Section] = field(default_factory=list)
    fulltext_source: str = ""  # URL the sections were read from; empty = abstract only
    reference_count: int | None = None

    # Reader state
    status: str = "To read"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    added_at: str = ""
    digests: dict = field(default_factory=dict)  # keyed by "<depth>|<audience>"

    # Semantic Scholar, best effort
    citation_count: int | None = None
    citations_checked: str = ""

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def year(self) -> int:
        return date.fromisoformat(self.published[:10]).year

    @property
    def has_full_text(self) -> bool:
        return bool(self.sections)

    @property
    def short_authors(self) -> str:
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."

    def full_text(self) -> str:
        return "\n\n".join(f"{s.title}\n{s.text}" for s in self.sections)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        d = dict(d)
        d["sections"] = [Section(**s) for s in d.get("sections", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
