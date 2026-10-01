from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings, get_settings


@lru_cache(maxsize=1)
def get_embeddings(settings: Settings | None = None) -> HuggingFaceEmbeddings:
    settings = settings or get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embed_model)


@lru_cache(maxsize=1)
def get_vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )


def _reset_vector_store_cache() -> None:
    get_vector_store.cache_clear()


def load_pdf_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    for path in pdf_paths:
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return docs


def build_index(rebuild: bool = False) -> dict[str, int]:
    settings = get_settings()
    pdf_paths = sorted(settings.data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {settings.data_dir}")

    if rebuild:
        store = get_vector_store(settings)
        store.delete_collection()
        _reset_vector_store_cache()

    docs = load_pdf_documents(settings.data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)

    store = get_vector_store(settings)
    if splits:
        store.add_documents(documents=splits)

    return {
        "pdfs": len(pdf_paths),
        "pages": len(docs),
        "chunks": len(splits),
    }


def index_chunk_count() -> int:
    settings = get_settings()
    if not settings.index_dir.exists():
        return 0
    try:
        return len(get_vector_store(settings).get(include=[])["ids"])
    except Exception:
        return 0


def ensure_index() -> None:
    if index_chunk_count() == 0:
        raise SystemExit(
            f"Index is empty (or missing) at {get_settings().index_dir}.\n"
            "Run: uv run pdf-rag index"
        )
