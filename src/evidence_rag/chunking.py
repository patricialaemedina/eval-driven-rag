import re
from pathlib import Path

from .models import Chunk

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(text: str, source: str = "document") -> list[Chunk]:
    """Split Markdown at headings while retaining each heading as retrieval context."""
    chunks: list[Chunk] = []
    heading = "Introduction"
    body: list[str] = []

    def flush() -> None:
        cleaned = "\n".join(body).strip()
        if not cleaned:
            return
        chunk_id = f"{source}#{len(chunks) + 1}"
        chunks.append(Chunk(id=chunk_id, heading=heading, text=cleaned))

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            heading = match.group(2)
            body = []
        else:
            body.append(line)
    flush()
    return chunks


def load_markdown(path: Path) -> list[Chunk]:
    return chunk_markdown(path.read_text(encoding="utf-8"), source=path.stem)
