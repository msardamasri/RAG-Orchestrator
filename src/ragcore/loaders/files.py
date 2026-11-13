"""Load documents from files: PDF (pypdf), Markdown and plain text."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ragcore.types import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


class FileLoader:
    """Every supported file under ``path`` (or the file itself). The document id is the path relative to the root."""

    def __init__(self, path: str | Path, metadata: dict | None = None):
        self.path = Path(path)
        self.metadata = metadata or {}

    def _files(self) -> Iterator[Path]:
        if self.path.is_file():
            yield self.path
            return
        for file in sorted(self.path.rglob("*")):
            if file.is_file() and file.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
                yield file

    def load(self) -> Iterable[Document]:
        root = self.path if self.path.is_dir() else self.path.parent
        for file in self._files():
            suffix = file.suffix.lower()
            text = read_pdf(file) if suffix == ".pdf" else file.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            doc_id = file.relative_to(root).as_posix()
            kind = "pdf" if suffix == ".pdf" else ("markdown" if suffix in {".md", ".markdown"} else "text")
            yield Document(id=doc_id, text=text, source=doc_id, metadata={"kind": kind, **self.metadata})
