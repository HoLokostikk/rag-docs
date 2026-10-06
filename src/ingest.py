
from pathlib import Path

import pymupdf4llm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def convert_all():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(RAW_DIR.glob("*.pdf")):
        out_path = PROCESSED_DIR / f"{path.stem}.md"
        if out_path.exists():
            print(f"skip {path.name}")
            continue

        print(f"converting {path.name}...", flush=True)
        markdown = pymupdf4llm.to_markdown(str(path))
        out_path.write_text(markdown, encoding="utf-8")
        print(f"  -> {out_path} ({len(markdown)} chars)")


if __name__ == "__main__":
    convert_all()