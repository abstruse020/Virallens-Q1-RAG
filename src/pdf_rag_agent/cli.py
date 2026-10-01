from __future__ import annotations

import argparse
import json
import re
import sys

USAGE = """\
pdf-rag — RAG agent over the PDFs in data/

usage:
  pdf-rag index [--rebuild]   Build (or rebuild) the vector index from data/*.pdf
  pdf-rag                     Interactive chat REPL
  pdf-rag <question...>       One-shot question
  pdf-rag -h | --help         Show this help
"""

ACTIVITY_HEADER = "─── agent activity ───"
ANSWER_HEADER = "─── answer ───"
RULE = "─" * 40

# Tool results whose content is pure line-number/file-open noise: collapse to a checkmark.
QUIET_RESULT_TOOLS = {"read_file", "glob", "ls", "grep", "edit_file", "write_file", "delete"}
# Tool calls worth showing with their arguments (others are collapsed when repeated).
VERBOSE_CALL_TOOLS = {"search_pdf_corpus"}


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def _error(text: str) -> str:
    return f"\033[31m{text}\033[0m" if sys.stdout.isatty() else text


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _first_line(content: str) -> str:
    content = content.strip()
    return content.splitlines()[0] if content else ""


def _run(agent, state: dict) -> dict:
    """Stream one agent turn, printing the activity section; return final state."""
    final: dict | None = None
    for namespace, mode, data in agent.stream(
        state,
        stream_mode=["updates", "values"],
        subgraphs=True,
    ):
        if mode == "values":
            final = data
            continue
        _render_updates(namespace, data)
    if final is None:
        final = agent.invoke(state)
    return final


def _render_updates(namespace, data) -> None:
    if not isinstance(data, dict):
        return
    # Only one subagent exists in this app; nested namespaces are its activity.
    prefix = "[chunk-analyst] " if namespace else ""
    for update in data.values():
        if not isinstance(update, dict):
            continue
        for line in _lines_for_messages(update.get("messages") or []):
            if namespace and _is_subagent_mechanics(line):
                continue
            rendered = f"{prefix}{line}"
            if "Error" in line or "error" in line:
                print(_error(rendered), flush=True)
            else:
                print(_dim(rendered), flush=True)


def _is_subagent_mechanics(line: str) -> bool:
    """Hide filesystem mechanics inside subagent turns; their summaries are shown instead."""
    if line.startswith("←") and "Error" in line:
        return False
    return bool(re.match(r"^(→ (read_file|glob|ls|grep|edit_file|write_file|delete)\b|✓ )", line))


def _lines_for_messages(messages) -> list[str]:
    """Turn a batch of stream messages into grouped, human-readable activity lines."""
    events: list[tuple[str, str, object]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        tool_calls = getattr(msg, "tool_calls", None) or []
        if msg_type == "ai" and tool_calls:
            for call in tool_calls:
                events.append(("call", call.get("name") or "?", call))
        elif msg_type == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            events.append(("result", getattr(msg, "name", "") or "?", content))

    lines: list[str] = []
    i = 0
    while i < len(events):
        kind, name, _ = events[i]
        j = i
        while j < len(events) and events[j][0] == kind and events[j][1] == name:
            j += 1
        group = events[i:j]
        if kind == "call":
            lines.extend(_call_lines(name, group))
        else:
            lines.extend(_result_lines(name, group))
        i = j
    return lines


def _call_lines(name: str, group: list) -> list[str]:
    if len(group) > 1:
        suffix = " (parallel)" if name == "task" else ""
        return [f"→ {name} ×{len(group)}{suffix}"]
    args = group[0][2].get("args", {}) or {}
    if name == "task":
        desc = args.get("description", "")
        return [f"→ task: {_short(desc, 70)}"]
    if name in VERBOSE_CALL_TOOLS:
        rendered = json.dumps(args, ensure_ascii=False)
        return [f"→ {name}({_short(rendered, 120)})"]
    return [f"→ {name}"]


def _result_lines(name: str, group: list) -> list[str]:
    contents = [entry[2] for entry in group]
    if name == "search_pdf_corpus":
        if len(contents) > 1:
            return [f"← search_pdf_corpus ×{len(contents)}"]
        content = contents[0]
        first = _first_line(content)
        match = re.search(r"(/retrieved/[0-9a-f]+/)", content)
        if match and "Saved" in first:
            count = re.search(r"(\d+)", first)
            label = f"{count.group(1)} chunks saved" if count else "chunks saved"
            return [f"← {label} → {match.group(1)}"]
        return [f"← {_short(first)}"]
    if name in QUIET_RESULT_TOOLS:
        quiet = [c for c in contents if not c.lstrip().startswith("Error")]
        errors = [c for c in contents if c.lstrip().startswith("Error")]
        lines = [f"✓ {name} ×{len(quiet)}"] if quiet else []
        lines += [f"← {name}: {_short(_first_line(e), 120)}" for e in errors]
        return lines
    # Subagent summaries (task results) and anything else: one line each.
    return [f"← {_short(_first_line(c))}" for c in contents if _first_line(c)]


def _final_answer(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            return msg.text if isinstance(msg.text, str) else str(msg.text)
    last = state.get("messages", [])[-1] if state.get("messages") else None
    if last is not None:
        return last.text if isinstance(last.text, str) else str(last)
    return ""


def _reflow(text: str) -> str:
    """Join soft-wrapped lines inside paragraphs; keep lists, headings, tables, code intact."""
    blocks = re.split(r"\n{2,}", text.strip())
    out: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        structural = any(
            re.match(r"\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|\||```)", line)
            for line in lines
        )
        if structural or "```" in block:
            out.append(block)
        else:
            joined = " ".join(line.strip() for line in lines if line.strip())
            out.append(re.sub(r" {2,}", " ", joined))
    return "\n\n".join(out)


def cmd_index(rebuild: bool) -> int:
    from .index import build_index, get_settings

    settings = get_settings()
    try:
        stats = build_index(rebuild=rebuild)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    verb = "Rebuilt" if rebuild else "Indexed"
    print(
        f"{verb} {stats['chunks']} chunks from {stats['pages']} pages "
        f"across {stats['pdfs']} PDFs -> {settings.index_dir}"
    )
    return 0


def cmd_ask(question: str) -> int:
    from .agent import build_agent

    agent = build_agent()
    print(_dim(ACTIVITY_HEADER), flush=True)
    state = _run(agent, {"messages": [{"role": "user", "content": question}]})
    answer = _final_answer(state)
    print(_dim(ANSWER_HEADER))
    if answer:
        print(_reflow(answer))
    print(_dim(RULE))
    return 0


def cmd_chat() -> int:
    from .agent import build_agent
    from .config import get_settings

    settings = get_settings()
    agent = build_agent()
    print(f"pdf-rag chat — model {settings.model_id}, index {settings.index_dir}")
    print("Commands: /new (reset), /quit (exit)\n")

    state: dict = {"messages": []}
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"/quit", "/exit", "exit", "quit"}:
            break
        if question == "/new":
            state = {"messages": []}
            print("(conversation reset)")
            continue
        print(_dim(ACTIVITY_HEADER), flush=True)
        try:
            state = _run(agent, {"messages": state["messages"] + [{"role": "user", "content": question}]})
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        answer = _final_answer(state)
        print(_dim(ANSWER_HEADER))
        if answer:
            print(_reflow(answer))
        print()
    return 0


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return
    if args and args[0] == "index":
        parser = argparse.ArgumentParser(prog="pdf-rag index", add_help=False)
        parser.add_argument("--rebuild", action="store_true")
        opts = parser.parse_args(args[1:])
        raise SystemExit(cmd_index(rebuild=opts.rebuild))
    if args and args[0] == "chat":
        raise SystemExit(cmd_chat())
    if args and args[0] == "ask":
        args = args[1:]
        if not args:
            print(USAGE, file=sys.stderr)
            raise SystemExit(2)
    if not args:
        raise SystemExit(cmd_chat())
    raise SystemExit(cmd_ask(" ".join(args)))


if __name__ == "__main__":
    main()
