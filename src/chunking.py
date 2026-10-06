import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text : str
    source : str
    index : int
    meta : dict = field(default_factory = dict)

    @property
    def char_len(self) -> int:
        return len(self.text)

def clean_text(text: str) -> str:
    text = re.sub(r"</?[a-z]+>", " ", text)
    text = re.sub(r"~~|__|\*\*|\*", "", text)
    text = re.sub(r"_([A-Za-z0-9]+)_", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def fixed_size(text : str, source : str, size : int = 800, overlap : int = 100) -> list[Chunk]:
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, source, index, {"strategy" : "fixed", "size" : size}))

            index += 1
        start = end - overlap

    return chunks

def recursive(text : str, source : str, size: int = 800, overlap : int = 100) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces = []
    for para in paragraphs:
        if len(para) < size:
            pieces.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) + 1 <= size:
                    buffer = f"{buffer} {sent}".strip()
                else:
                    if buffer:
                        pieces.append(buffer)
                    buffer = sent
            if buffer:
                pieces.append(buffer)

    chunks = []
    buffer = ""
    index = 0
    for piece in pieces:
        if len(buffer) +len(piece) + 1 <= size:
            buffer = f"{buffer}\n\n{piece}".strip()
        else:
            if buffer:
                chunks.append(Chunk(buffer, source, index,
                                    {"strategy": "recursive", "size": size}))
                index += 1
            buffer = piece

    if buffer:
        chunks.append(Chunk(buffer, source, index,
                            {"strategy": "recursive", "size": size}))

    return chunks


def by_headings(text: str, source: str, max_size: int = 1500) -> list[Chunk]:
    lines = text.split("\n")
    sections = []
    current_heading = ""
    buffer = []

    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            if buffer:
                sections.append((current_heading, "\n".join(buffer)))
            current_heading = line.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer)))

    chunks = []
    index = 0
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue

        full = f"# {heading}\n\n{body}" if heading else body

        if len(full) <= max_size:
            chunks.append(Chunk(full, source, index,
                                {"strategy": "headings", "heading": heading}))
            index += 1
        else:
            for sub in recursive(body, source, size=max_size):
                sub.text = f"# {heading}\n\n{sub.text}" if heading else sub.text
                sub.index = index
                sub.meta = {"strategy": "headings", "heading": heading, "split": True}
                chunks.append(sub)
                index += 1

    return chunks


STRATEGIES = {
    "fixed": fixed_size,
    "recursive": recursive,
    "headings": by_headings,
}

NOISE_HEADINGS = {
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "appendix", "author contributions",
}


def is_noise(chunk: Chunk) -> bool:
    heading = chunk.meta.get("heading", "").lower().strip()

    if any(h in heading for h in NOISE_HEADINGS):
        return True

    year_lines = len(re.findall(r"(19|20)\d{2}[a-z]?\.", chunk.text))
    if year_lines >= 5:
        return True

    return False