# PDF RAG Agent — multi-agent RAG over local PDFs

A **multi-agent RAG** agent over the PDFs in `data/`. A LangGraph
orchestrator plans searches, delegates every retrieved chunk to a
`chunk-analyst` subagent, then synthesizes an answer with inline citations.

- Agent harness: **deepagents** (`create_deep_agent`) — compiles to a LangGraph graph
- LLM provider: **OpenRouter** via `langchain-openrouter` (default `openai/gpt-4o-mini`)
- Index: local `sentence-transformers/all-MiniLM-L6-v2` embeddings → Chroma in `.rag_index/`

## Agent roles

### 1. Orchestrator (main agent)

Defined in `src/pdf_rag_agent/agent.py` with the system prompt built in
`src/pdf_rag_agent/prompts.py`. It owns the whole conversation and never reads
chunk files itself:

- **Plan** — break the question into focused search queries
- **Search** — call `search_pdf_corpus`, which writes matches into the agent's
  virtual filesystem under `/retrieved/<batch>/chunk_N.md`
- **Delegate** — launch one `task()` per chunk file in parallel (up to
  `PDF_RAG_MAX_ANALYSTS`, default 3), copying the returned path verbatim
- **Synthesize** — merge subagent summaries into one answer, citing
  `[filename.pdf, p.N]`
- **Verify** — only when the returned summaries don't fully answer the question:
  re-search with a refined query (max 2 rounds), otherwise answer from the best
  evidence gathered or say the corpus doesn't cover it

### 2. `chunk-analyst` (the subagent in use)

Registered via `subagents=[chunk_analyst_subagent()]` in
`src/pdf_rag_agent/agent.py`. One instance runs per retrieved chunk:

- reads exactly one file path with `read_file` 
- extracts facts, claims, figures, dates relevant to the question
- returns a summary under 300 words plus the chunk header line (filename + page)

It inherits the parent's toolset, but its prompt only sanctions `read_file` —
searching and further delegation stay with the orchestrator.

### Tools

| Tool | Owner | Purpose |
| --- | --- | --- |
| `search_pdf_corpus` | orchestrator | Chroma top-k search, saves chunks to the virtual FS |
| `task` | orchestrator | spawns the `chunk-analyst` subagent |
| `read_file`, `ls`, `glob`, `grep`, `write_file`, `edit_file` | both | virtual filesystem over `StateBackend` |

## LangGraph flow + Deepagents flow

### Deepagents flow

`create_deep_agent(...)` (`src/pdf_rag_agent/agent.py`) assembles the harness as
a middleware stack wrapped around a single model:

- **FilesystemMiddleware** — the `StateBackend` virtual FS the search tool writes into
- **SubAgentMiddleware** — exposes the `task` tool and builds `chunk-analyst` as its own agent
- **summarization / patch-tool-calls / unsupported-content** — context and message hygiene

The orchestrator prompt (`RAG_WORKFLOW_INSTRUCTIONS` + `SUBAGENT_DELEGATION_INSTRUCTIONS`)
carries the plan/delegate/synthesize/verify policy; the subagent prompt
(`CHUNK_ANALYST_INSTRUCTIONS`) carries the per-chunk extraction policy.

### LangGraph flow

`create_deep_agent` calls `langchain.agents.create_agent` and returns a compiled
LangGraph `CompiledStateGraph` (`recursion_limit` 9999). One turn runs like this:

```text
user question
  → orchestrator (LangGraph turn)
      1. plan          break the question into search queries
      2. search        search_pdf_corpus → chunks saved to /retrieved/
      3. delegate      task() per chunk → chunk-analyst reads the file,
                                       returns a ≤300-word summary
      4. synthesize    merge summaries → answer with [file.pdf, p.N] citations
      5. verify        re-search only if the summaries don't fully answer
                       (max 2 rounds), then steps 2–4 again
  → final answer
```

The `chunk-analyst` step is a nested LangGraph subgraph, so the tool →
model loop runs twice: once for the orchestrator, once inside each `task()`.
The CLI streams with `stream_mode=["updates", "values"], subgraphs=True`
(`src/pdf_rag_agent/cli.py`) and prefixes nested activity with `[chunk-analyst]`,
so both agents' steps show up in one activity feed.

## Approach followed

1. **Ingest** — `pypdf` extracts text page by page (metadata: source file + page number).
2. **Chunk** — `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap, start offsets).
3. **Embed & store** — local HuggingFace `all-MiniLM-L6-v2` vectors into a persistent
   Chroma collection at `.rag_index/`; `pdf-rag index [--rebuild]` builds it.
4. **Retrieve** — `search_pdf_corpus(query)` runs similarity search (top-k 4) and drops
   the chunks as markdown files into the agent filesystem instead of into the prompt.
5. **Fan-out** — parallel `chunk-analyst` tasks read those files, keeping large raw
   chunk text out of the orchestrator's context.
6. **Synthesize + harden** — answers cite sources inline; retrieved PDF text is treated
   as data only (embedded instructions in chunk content are ignored), and the agent may
   not answer from memory when corpus evidence is required.

## OpenRouter

All chat completions go through **OpenRouter** (`langchain-openrouter`,
`init_chat_model` with the `openrouter:` prefix). Set the key in `.env`:

```bash
cp .env.example .env
# OPENROUTER_API_KEY=sk-or-...
# PDF_RAG_MODEL=openai/gpt-4o-mini   # any tool-calling model on OpenRouter
```

## Run locally

```bash
uv sync
uv run pdf-rag index          # build the vector index from data/*.pdf
uv run pdf-rag                # interactive chat
uv run pdf-rag "What is X?"    # one-shot question
```

## Run with Docker

The image bakes the corpus, the embedding model, and a prebuilt index, so it runs
out of the box. The API key is passed at runtime (never baked into the image):

```bash
docker build --allow network.host -t pdf-rag .
```

`--allow network.host` is required. The network-heavy build steps (`uv sync`,
embedding model download) declare `RUN --network=host` because Docker's default
bridge DNS (`127.0.0.11` → `8.8.8.8`) fails under uv's parallel downloads with
`dns error … Try again`, and host networking is a privileged BuildKit
entitlement — without the flag the build stops with
`network.host is not allowed`. Drop either piece and the build fails.

```bash
docker run --rm -it \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  pdf-rag "What does the complaint allege?"

docker run --rm -it -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY pdf-rag   # chat REPL
```

Use your own corpus or keep the index across containers (the index lives in the
container layer, so mount `/app/.rag_index` to survive `docker run --rm`):

```bash
docker run --rm -it \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v "$PWD/data:/app/data" \
  -v pdf-rag-index:/app/.rag_index \
  pdf-rag index --rebuild

docker run --rm -it \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v "$PWD/data:/app/data" \
  -v pdf-rag-index:/app/.rag_index \
  pdf-rag "Your question"
```

> **CPU-only torch.** `pyproject.toml` pins `torch` to the CPU build from
> `download.pytorch.org/whl/cpu`, so no `nvidia-*` wheels are installed. The
> agent never uses a GPU anyway — local embeddings run on CPU and chat runs on
> OpenRouter. The image is ~2.8 GB (it was ~10.4 GB with the CUDA build).
> To switch back to CUDA: delete the `[[tool.uv.index]]` / `[tool.uv.sources]`
> blocks from `pyproject.toml` and re-run `uv lock`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | required for chat |
| `PDF_RAG_MODEL` | `openai/gpt-4o-mini` | OpenRouter model id |
| `PDF_RAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model |
| `PDF_RAG_CHUNK_SIZE` / `PDF_RAG_CHUNK_OVERLAP` | `1000` / `200` | splitter settings |
| `PDF_RAG_TOP_K` | `4` | chunks per search |
| `PDF_RAG_MAX_ANALYSTS` | `3` | parallel `chunk-analyst` tasks per round |
| `PDF_RAG_DATA_DIR` / `PDF_RAG_INDEX_DIR` | `./data` / `./.rag_index` | corpus and index paths |
