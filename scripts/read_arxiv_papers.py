"""Download arXiv papers and dump their text, for targeted overlap checks.

    python scripts/read_arxiv_papers.py --ids 2407.01032,2506.18162

Writes paper/related/<id>.pdf and <id>.txt, then prints the first `--head` characters
of each paper (title/abstract area) so a human can judge whether a claim overlaps.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "related"


def download(arxiv_id: str, target: Path) -> Path | None:
    if target.exists() and target.stat().st_size > 10_000:
        return target
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=90, headers={"User-Agent": "paper-reader/1.0"})
            response.raise_for_status()
            if response.content[:4] != b"%PDF":
                raise RuntimeError(f"not a PDF (first bytes: {response.content[:16]!r})")
            target.write_bytes(response.content)
            return target
        except Exception as exc:  # noqa: BLE001
            print(f"  download attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
            time.sleep(4)
    return None


def extract(pdf_path: Path, text_path: Path) -> bool:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed: pip install pypdf")
        return False
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            chunks.append(f"[page extraction failed: {exc}]")
    text_path.write_text("\n".join(chunks), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, help="comma-separated arXiv ids")
    parser.add_argument("--head", type=int, default=1800, help="characters of each paper to print")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arxiv_id in [a.strip() for a in args.ids.split(",") if a.strip()]:
        pdf_path = OUT_DIR / f"{arxiv_id}.pdf"
        text_path = OUT_DIR / f"{arxiv_id}.txt"
        print(f"=== {arxiv_id} ===")
        if download(arxiv_id, pdf_path) is None:
            print("  download FAILED")
            continue
        size_kb = pdf_path.stat().st_size / 1024
        if not text_path.exists() or text_path.stat().st_size == 0:
            if not extract(pdf_path, text_path):
                continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        print(f"  pdf {size_kb:.0f} KB | text {len(text):,} chars | pages ~{len(text) // 2500}")
        head = " ".join(text[: args.head].split())
        print(f"  HEAD: {head}\n")
    print(f"artifacts in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
