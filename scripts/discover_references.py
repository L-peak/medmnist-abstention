"""Find candidate references on a topic via the arXiv API.

    python scripts/discover_references.py --query 'all:"conformal prediction" AND all:"medical imaging"'

Prints year | arXiv id | title | journal-ref for the top hits so a human can pick
which ones to add to verify_references.py.
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET

import requests

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search(query: str, limit: int = 8, sort: str = "relevance") -> list[dict]:
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": query, "max_results": limit, "sortBy": sort},
        timeout=45,
        headers={"User-Agent": "reference-discovery/1.0"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    results = []
    for entry in root.findall("a:entry", NS):
        results.append({
            "id": (entry.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1],
            "title": " ".join((entry.findtext("a:title", "", NS) or "").split()),
            "year": (entry.findtext("a:published", "", NS) or "")[:4],
            "authors": [a.findtext("a:name", "", NS) for a in entry.findall("a:author", NS)][:3],
            "journal_ref": entry.findtext("arxiv:journal_ref", "", NS) or "",
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sort", default="relevance", choices=["relevance", "lastUpdatedDate", "submittedDate"])
    parser.add_argument("--repeat", type=int, default=1, help="repeat the same query (arXiv rate limit)")
    args = parser.parse_args()

    for run in range(args.repeat):
        if run:
            time.sleep(3)
        print(f"\n=== query: {args.query} (sort={args.sort}) ===")
        for hit in search(args.query, args.limit, args.sort):
            authors = ", ".join(hit["authors"])
            print(f"{hit['year']} | arXiv:{hit['id']} | {hit['title'][:90]}")
            print(f"        authors: {authors[:80]}")
            if hit["journal_ref"]:
                print(f"        journal-ref: {hit['journal_ref'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
