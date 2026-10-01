from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Keep stderr quiet while loading embedding models (noise, not signal).
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


def _find_root() -> Path:
    env_root = os.environ.get("PDF_RAG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if any(candidate.joinpath("data").glob("*.pdf")):
            return candidate
    return cwd


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    index_dir: Path
    collection_name: str
    model: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_concurrent_analysts: int

    @property
    def model_id(self) -> str:
        model = self.model.strip()
        return model if model.startswith("openrouter:") else f"openrouter:{model}"

    @property
    def api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY") or None

    def require_api_key(self) -> None:
        if not self.api_key:
            sys.exit(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://openrouter.ai/keys)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _find_root()
    load_dotenv(root / ".env")
    return Settings(
        root=root,
        data_dir=Path(os.environ.get("PDF_RAG_DATA_DIR", root / "data")).expanduser().resolve(),
        index_dir=Path(os.environ.get("PDF_RAG_INDEX_DIR", root / ".rag_index")).expanduser().resolve(),
        collection_name=os.environ.get("PDF_RAG_COLLECTION", "pdf_rag"),
        model=os.environ.get("PDF_RAG_MODEL", "openai/gpt-4o-mini"),
        embed_model=os.environ.get("PDF_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=int(os.environ.get("PDF_RAG_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.environ.get("PDF_RAG_CHUNK_OVERLAP", "200")),
        top_k=int(os.environ.get("PDF_RAG_TOP_K", "4")),
        max_concurrent_analysts=int(os.environ.get("PDF_RAG_MAX_ANALYSTS", "3")),
    )
