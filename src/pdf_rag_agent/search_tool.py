from __future__ import annotations

import uuid
from functools import lru_cache

from deepagents.backends import StateBackend
from langchain.tools import BaseTool, tool

from .index import get_vector_store


@lru_cache(maxsize=1)
def get_backend() -> StateBackend:
    return StateBackend()


@lru_cache(maxsize=1)
def get_search_tool() -> BaseTool:
    @tool(parse_docstring=True)
    def search_pdf_corpus(query: str) -> str:
        """Search the indexed PDF corpus and save matching chunks to the agent filesystem.

        Args:
            query: Natural language search query.

        Returns:
            File paths where retrieved chunks were saved under /retrieved/.
        """
        from .config import get_settings

        settings = get_settings()
        vector_store = get_vector_store(settings)
        retrieved_docs = vector_store.similarity_search(query, k=settings.top_k)
        if not retrieved_docs:
            return "No matching chunks found in the PDF corpus."

        batch_id = uuid.uuid4().hex[:8]
        uploads: list[tuple[str, bytes]] = []
        saved_paths: list[str] = []
        for index, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            path = f"/retrieved/{batch_id}/chunk_{index}.md"
            content = f"# Source: {source} | page {page}\n\n{doc.page_content}\n"
            uploads.append((path, content.encode("utf-8")))
            saved_paths.append(path)

        get_backend().upload_files(uploads)
        return (
            f"Saved {len(saved_paths)} PDF chunks:\n" + "\n".join(saved_paths)
        )

    return search_pdf_corpus
