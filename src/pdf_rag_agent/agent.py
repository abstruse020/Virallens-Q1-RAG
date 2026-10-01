from __future__ import annotations

from functools import lru_cache

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from .config import get_settings
from .index import ensure_index
from .prompts import chunk_analyst_subagent, get_instructions
from .search_tool import get_backend, get_search_tool


@lru_cache(maxsize=1)
def build_agent():
    settings = get_settings()
    settings.require_api_key()
    ensure_index()

    model = init_chat_model(model=settings.model_id)
    return create_deep_agent(
        model=model,
        tools=[get_search_tool()],
        backend=get_backend(),
        system_prompt=get_instructions(),
        subagents=[chunk_analyst_subagent()],
    )
