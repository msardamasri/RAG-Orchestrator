"""Command line: ``ragcore ingest | ask | chat | eval``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragcore.config import RagConfig, build_rag, load_profile
from ragcore.pipeline.rag import Rag
from ragcore.types import Answer, ConversationState

DEFAULT_PROFILE = "profiles/local.toml"


def _config(args: argparse.Namespace) -> RagConfig:
    path = Path(args.profile)
    if not path.exists():
        if args.profile != DEFAULT_PROFILE:
            sys.exit(f"profile not found: {path}")
        config = RagConfig()
    else:
        config = load_profile(path)
    if getattr(args, "store", None):
        config.store = {**config.store, "path": args.store}
    return config


def _build(args: argparse.Namespace, need_generator: bool = False) -> Rag:
    config = _config(args)
    if not need_generator:
        config.generator = None
    if getattr(args, "no_rerank", False):
        config.reranker = {"kind": "none"}
    rag = build_rag(config)
    if need_generator and rag.generator is None:
        sys.exit("this profile has no [generator]; add one or use a command that only retrieves")
    available = getattr(rag.generator, "available", None)
    if need_generator and available is not None and not available():
        sys.exit(f"the generator at {rag.generator.base_url} is not answering. Start your model server first "
                 "(for the local profile: `brain model serve` in personal-brain, or any OpenAI-style server).")
    return rag


def _print_answer(answer: Answer, show_sources: bool = True) -> None:
    if answer.standalone_question and answer.standalone_question != answer.question:
        print(f"  (searched for: {answer.standalone_question})")
    if answer.topic_switched:
        print("  (new topic)")
    print(answer.text)
    if show_sources and answer.citations:
        print()
        for citation in answer.citations:
            hit = answer.hits[citation.n - 1]
            section = hit.chunk.metadata.get("section")
            print(f"  [{citation.n}] {citation.source}" + (f" · {section}" if section else ""))
    elif answer.abstained and answer.reason:
        print(f"  (refused: {answer.reason.replace('_', ' ')})")


def cmd_ingest(args: argparse.Namespace) -> None:
    from ragcore.loaders.files import FileLoader

    rag = _build(args)
    report = rag.ingest(FileLoader(args.path), prune=args.prune)
    print(f"{report.documents} documents: {report.ingested} ingested ({report.chunks} chunks), "
          f"{report.skipped} unchanged, {report.removed} removed. Store now holds {rag.store.count()} chunks.")
    for doc_id in report.empty:
        hint = " (a scanned PDF? it needs OCR first)" if doc_id.lower().endswith(".pdf") else ""
        print(f"  no text in {doc_id}{hint}")
    for doc_id, reason in report.failed.items():
        print(f"  could not read {doc_id}: {reason} (what was stored for it before is kept)")


def cmd_search(args: argparse.Namespace) -> None:
    rag = _build(args)
    for n, hit in enumerate(rag.retrieve(args.query, mode=args.mode), start=1):
        scores = " ".join(f"{k}={v:.3f}" for k, v in hit.scores.items())
        text = " ".join(hit.chunk.text.split())
        print(f"{n}. {hit.chunk.doc_id}  [{scores}]\n   {text[:220]}")


def cmd_ask(args: argparse.Namespace) -> None:
    _print_answer(_build(args, need_generator=True).ask(args.question))


def cmd_chat(args: argparse.Namespace) -> None:
    rag = _build(args, need_generator=True)
    state = ConversationState()
    print("Ask a question. Empty line or Ctrl-D to quit.")
    while True:
        try:
            message = input("\n> ").strip()
        except EOFError:
            break
        if not message:
            break
        _print_answer(rag.chat(state, message))


def cmd_eval(args: argparse.Namespace) -> None:
    from ragcore.eval.dataset import load_conversations, load_questions
    from ragcore.eval.runner import format_report, run, save_report
    from ragcore.loaders.files import FileLoader

    data = Path(args.data)
    config = _config(args)
    config.store = {"kind": "sqlite", "path": ":memory:"} if config.store.get("kind") == "sqlite" else config.store
    if not (args.generate or args.rewrite):
        config.generator = None
    rag = build_rag(config)
    available = getattr(rag.generator, "available", None)
    if rag.generator is not None and available is not None and not available():
        sys.exit(f"the generator at {rag.generator.base_url} is not answering; run without --generate/--rewrite "
                 "or start your model server")
    report_ingest = rag.ingest(FileLoader(data / "corpus"))
    print(f"Ingested {report_ingest.documents} documents into {report_ingest.chunks} chunks.\n")
    questions = load_questions(data / "questions.jsonl")
    conversations_path = data / "conversations.jsonl"
    conversations = load_conversations(conversations_path) if conversations_path.exists() else []
    report = run(rag, questions, conversations, generate=args.generate)
    print(format_report(report))
    if not args.no_save:
        print(f"\nSaved to {save_report(report, args.out)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragcore", description="A small, pluggable RAG baseline.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="TOML profile naming every part")
    parser.add_argument("--store", help="override the store path of the profile")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="load, chunk, embed and store a file or folder")
    ingest.add_argument("path")
    ingest.add_argument("--prune", action="store_true", help="remove stored documents that are not under PATH")
    ingest.set_defaults(run=cmd_ingest)

    search = commands.add_parser("search", help="show the passages retrieved for a query (no model needed)")
    search.add_argument("query")
    search.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    search.add_argument("--no-rerank", action="store_true")
    search.set_defaults(run=cmd_search)

    ask = commands.add_parser("ask", help="answer one question with citations")
    ask.add_argument("question")
    ask.set_defaults(run=cmd_ask)

    chat = commands.add_parser("chat", help="multi-turn conversation in the terminal")
    chat.set_defaults(run=cmd_chat)

    evaluate = commands.add_parser("eval", aliases=["evaluate"],
                                   help="measure retrieval, refusals, conversations and answers")
    evaluate.add_argument("--data", default="eval_data", help="folder with corpus/ and the two .jsonl files")
    evaluate.add_argument("--rewrite", action="store_true", help="also test the model's follow-up rewrites")
    evaluate.add_argument("--generate", action="store_true", help="also generate and check answers (slow)")
    evaluate.add_argument("--out", default="eval_results")
    evaluate.add_argument("--no-save", action="store_true")
    evaluate.set_defaults(run=cmd_eval)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
