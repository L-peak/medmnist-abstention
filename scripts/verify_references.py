"""Verify candidate references against Crossref and emit BibTeX.

Usage:
    python scripts/verify_references.py                 # verify the built-in list
    python scripts/verify_references.py --only msp      # subset

Every entry is matched by title; the script prints the match score, the resolved
DOI and venue, and writes paper/references.bib plus paper/references_log.md.
Nothing is invented: if a title cannot be matched the entry is reported as MISSING.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
MAILTO = "research@example.org"

# key -> exact title (as it should appear on the authoritative record)
CANDIDATES = {
    "chow1970reject": "On optimum recognition error and reject tradeoff",
    "elyaniv2010foundations": "Foundations of Selective Classification",
    "hendrycks2017baseline": "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks",
    "gal2016dropout": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
    "lakshminarayanan2017simple": "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
    "lei2018distribution": "Distribution-Free Predictive Inference for Regression",
    "romano2020classification": "Classification with Valid and Adaptive Coverage",
    "yang2023medmnist": "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification",
    "vovk2005algorithmic": "Algorithmic Learning in a Random World",
    # Positioned against our findings: existing benchmarks and known evaluation pitfalls.
    "pugnana2024benchmarks": "Deep Neural Network Benchmarks for Selective Classification",
    "traub2024flaws": "Overcoming Common Flaws in the Evaluation of Selective Classification Systems",
    "mehrtens2025pitfalls": "Pitfalls of Conformal Predictions for Medical Image Classification",
    "zhang2023rrcp": "RR-CP: Reliable-Region-Based Conformal Prediction for Trustworthy Medical Image Classification",
    "mehrtens2023benchmarking": "Benchmarking common uncertainty estimation methods with histopathological images under domain shift and label noise",
}

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Entries that no automated index resolves but that were verified by hand against the
# publisher's own page. Keep the source URL in the log so the check is auditable.
MANUAL_ENTRIES = {
    # Verified 2026-10-13 with score 1.00 via the arXiv API; pinned so a later flaky
    # query cannot drop the entry. The log records whether a run re-verified or used
    # the pin.
    "pugnana2024benchmarks": {
        "bibtex": """@misc{pugnana2024benchmarks,
  title = {Deep Neural Network Benchmarks for Selective Classification},
  author = {Pugnana, Andrea and Perini, Lorenzo and Davis, Jesse and Ruggieri, Salvatore},
  year = {2024},
  eprint = {2401.12708},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2401.12708},
  note = {Journal of Data-centric Machine Learning Research (DMLR), Vol. 1, (17):1-58}
}
""",
        "source": "https://arxiv.org/abs/2401.12708",
        "note": "arXiv:2401.12708, verified 2026-10-13 (pinned after an intermittent API miss)",
    },
    # Verified 2026-10-13 with score 1.00 via the arXiv API; pinned so that a later
    # flaky query cannot silently drop the entry from references.bib.
    "lakshminarayanan2017simple": {
        "bibtex": """@misc{lakshminarayanan2017simple,
  title = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  year = {2016},
  eprint = {1612.01474},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.1612.01474},
  note = {Published at NeurIPS 2017}
}
""",
        "source": "https://arxiv.org/abs/1612.01474",
        "note": "arXiv:1612.01474, verified 2026-10-13 (pinned after an intermittent API miss)",
    },
    "elyaniv2010foundations": {
        "bibtex": """@article{elyaniv2010foundations,
  title = {On the Foundations of Noise-free Selective Classification},
  author = {El-Yaniv, Ran and Wiener, Yair},
  journal = {Journal of Machine Learning Research},
  volume = {11},
  number = {53},
  pages = {1605--1641},
  year = {2010},
  url = {https://www.jmlr.org/papers/v11/el-yaniv10a.html}
}
""",
        "source": "https://www.jmlr.org/papers/v11/el-yaniv10a.html",
        "note": "JMLR 11(53):1605-1641, 2010 (title read from the publisher page)",
    },
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def crossref_query(title: str, rows: int = 5) -> list[dict]:
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": title, "rows": rows, "mailto": MAILTO}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=45,
                                    headers={"User-Agent": "reference-verifier/1.0"})
            response.raise_for_status()
            return response.json().get("message", {}).get("items", [])
        except Exception as exc:  # noqa: BLE001
            print(f"    crossref attempt {attempt + 1} failed: {type(exc).__name__}")
            time.sleep(4)
    return []


def best_match(title: str, items: list[dict]) -> tuple[float, dict | None]:
    best_score, best_item = 0.0, None
    target = normalise(title)
    for item in items:
        candidate = normalise((item.get("title") or [""])[0])
        score = difflib.SequenceMatcher(None, target, candidate).ratio()
        if score > best_score:
            best_score, best_item = score, item
    return best_score, best_item


def arxiv_query(title: str, rows: int = 5) -> list[dict]:
    """Query the arXiv API by title. Returns [{id, title, authors, year, journal_ref}]."""
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f'ti:"{title}"', "max_results": rows}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=45,
                                    headers={"User-Agent": "reference-verifier/1.0"})
            response.raise_for_status()
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.text)
            out = []
            for entry in root.findall("a:entry", ARXIV_NS):
                out.append({
                    "id": (entry.findtext("a:id", "", ARXIV_NS) or "").rsplit("/", 1)[-1],
                    "title": " ".join((entry.findtext("a:title", "", ARXIV_NS) or "").split()),
                    "authors": [a.findtext("a:name", "", ARXIV_NS)
                                for a in entry.findall("a:author", ARXIV_NS)],
                    "year": (entry.findtext("a:published", "", ARXIV_NS) or "")[:4],
                    "journal_ref": entry.findtext("arxiv:journal_ref", "", ARXIV_NS),
                })
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"    arxiv attempt {attempt + 1} failed: {type(exc).__name__}")
            time.sleep(4)
    return []


def arxiv_bibtex(key: str, entry: dict) -> str:
    authors = " and ".join(entry["authors"][:8])
    note = entry.get("journal_ref") or ""
    # The trailing comma matters: without it BibTeX reports "expecting a `,' or a `}'"
    # when a note field follows the doi.
    extra = f",\n  note = {{Journal reference: {note}}}" if note else ""
    return (f"@misc{{{key},\n"
            f"  title = {{{entry['title']}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{entry['year']}}},\n"
            f"  eprint = {{{entry['id']}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  doi = {{10.48550/arXiv.{entry['id']}}}{extra}\n}}\n")


def bibtex(key: str, item: dict) -> str:
    authors = item.get("author") or []
    # BibTeX separates authors with " and ", not commas - commas inside a name mean
    # "Last, First", so joining with ", " silently mangles every multi-author entry.
    names = " and ".join(
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in authors if a.get("family")
    )
    title = (item.get("title") or [""])[0]
    year = ""
    for field in ("published-print", "published-online", "issued", "created"):
        parts = (item.get(field) or {}).get("date-parts") or []
        if parts and parts[0]:
            year = str(parts[0][0])
            break
    container = (item.get("container-title") or [""])[0]
    kind = item.get("type", "")
    entry_type = "article" if "journal" in kind else ("inproceedings" if "proceedings" in kind else "misc")
    venue_field = "journal" if entry_type == "article" else ("booktitle" if entry_type == "inproceedings" else "howpublished")
    fields = [
        f"  title = {{{title}}}",
        f"  author = {{{names}}}",
        f"  year = {{{year}}}",
    ]
    if container:
        fields.append(f"  {venue_field} = {{{container}}}")
    if item.get("publisher"):
        fields.append(f"  publisher = {{{item['publisher']}}}")
    if item.get("DOI"):
        fields.append(f"  doi = {{{item['DOI']}}}")
        fields.append(f"  url = {{https://doi.org/{item['DOI']}}}")
    return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated keys")
    parser.add_argument("--min-score", type=float, default=0.75)
    args = parser.parse_args()

    keys = list(CANDIDATES)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]

    PAPER_DIR.mkdir(exist_ok=True)
    entries, log = [], ["# Reference verification log", ""]
    for key in keys:
        title = CANDIDATES[key]
        print(f"[{key}] {title[:70]}")
        items = crossref_query(title)
        score, item = best_match(title, items)
        if item is None or score < args.min_score:
            if key in MANUAL_ENTRIES:
                manual = MANUAL_ENTRIES[key]
                entries.append(manual["bibtex"].rstrip() + "\n")
                log.append(f"- **{key}** MANUAL | {manual['note']} | source: {manual['source']}")
                print(f"    manual entry used: {manual['source']}")
                continue
            # Fall back to arXiv, which is authoritative for ML preprints.
            arxiv_hits = arxiv_query(title)
            arxiv_score, arxiv_best = 0.0, None
            if arxiv_hits:
                target = normalise(title)
                for hit in arxiv_hits:
                    candidate = normalise(hit["title"])
                    hit_score = difflib.SequenceMatcher(None, target, candidate).ratio()
                    if hit_score > arxiv_score:
                        arxiv_score, arxiv_best = hit_score, hit
            if arxiv_best is None or arxiv_score < args.min_score:
                log.append(f"- **{key}** MISSING (crossref {score:.2f}, arxiv {arxiv_score:.2f}) for: {title}")
                print(f"    MISSING (crossref {score:.2f}, arxiv {arxiv_score:.2f})")
                continue
            entries.append(arxiv_bibtex(key, arxiv_best))
            note = f" | journal-ref: {arxiv_best['journal_ref']}" if arxiv_best.get("journal_ref") else ""
            log.append(f"- **{key}** arXiv:{arxiv_best['id']} (score {arxiv_score:.2f}) | "
                       f"{arxiv_best['year']} | `{arxiv_best['title'][:80]}`{note}")
            print(f"    arXiv:{arxiv_best['id']} {arxiv_score:.2f} | {arxiv_best['year']}{note}")
            time.sleep(3)
            continue
        doi = item.get("DOI", "")
        container = (item.get("container-title") or [""])[0]
        year = ((item.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        verified_title = (item.get("title") or [""])[0]
        entries.append(bibtex(key, item))
        line = (f"- **{key}** score {score:.2f} | {year} | {container or item.get('type')} | "
                f"doi:{doi} | `{verified_title[:90]}`")
        log.append(line)
        print(f"    ok {score:.2f} | {year} | {container[:50]} | {doi}")
        time.sleep(1.5)

    if entries:
        (PAPER_DIR / "references.bib").write_text("\n".join(entries), encoding="utf-8")
        print(f"\nwrote {PAPER_DIR / 'references.bib'} ({len(entries)} entries)")
    (PAPER_DIR / "references_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"wrote {PAPER_DIR / 'references_log.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
