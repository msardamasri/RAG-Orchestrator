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
    """Every supported file under ``path`` (or the file itself). The document id is the path relative to the root.

    Hidden files and folders (``.git``, ``.venv``…) are skipped. A file that cannot be read does not stop the
    load: it is listed in ``failed`` (id → reason) and left out, and ingest keeps what it stored for it before.
    A file with no text (a scanned PDF, an empty note) is still yielded, so ingest reports it as empty.
    """

    def __init__(self, path: str | Path, metadata: dict | None = None):
        self.path = Path(path)
        self.metadata = metadata or {}
        self.failed: dict[str, str] = {}

    def _root(self) -> Path:
        return self.path if self.path.is_dir() else self.path.parent

    def _files(self) -> Iterator[Path]:
        if self.path.is_file():
            yield self.path
            return
        for file in sorted(self.path.rglob("*")):
            relative = file.relative_to(self.path)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if file.is_file() and file.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
                yield file

    def load(self) -> Iterable[Document]:
        self.failed.clear()          # cleared, not replaced: callers may hold this dict while the load runs
        root = self._root()
        for file in self._files():
            doc_id = file.relative_to(root).as_posix()
            suffix = file.suffix.lower()
            try:
                # utf-8-sig drops the byte-order mark some editors write, which would hide a first "# Title".
                text = read_pdf(file) if suffix == ".pdf" else file.read_text(encoding="utf-8-sig", errors="replace")
            except Exception as error:          # a corrupt or unreadable file must not abort the whole load
                self.failed[doc_id] = f"{type(error).__name__}: {error}"
                continue
            kind = "pdf" if suffix == ".pdf" else ("markdown" if suffix in {".md", ".markdown"} else "text")
            yield Document(id=doc_id, text=text, source=doc_id, metadata={"kind": kind, **self.metadata})
