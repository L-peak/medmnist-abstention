"""Structural submission audit for the LaTeX manuscript (Elsevier-style checks).

    python scripts/submission_audit.py

This covers the parts of a journal's technical check that can be verified without the
journal's own web page: required sections and declarations, figure/table caption <-> text
reference consistency, highlights limits, abstract/keyword counts, leftover placeholders
and reference hygiene. It does not replace reading the target journal's live Guide for
Authors (abstract limits, article types, suggested reviewers, required upload files).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'ok' if ok else 'FIX'}] {name}{(' -- ' + detail) if detail else ''}")


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    body = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]

    # --- front matter -------------------------------------------------------
    check("title page carries author, affiliation, corresponding e-mail",
          all(token in tex for token in ("\\author[1]", "\\address[1]", "\\ead{")))
    # word count as a reader sees it: drop LaTeX commands, collapse math and dashes
    plain = re.sub(r"\$[^$]*\$", " X ", body)
    plain = re.sub(r"\\[a-zA-Z]+\*?", " ", plain)
    plain = plain.replace("---", " ").replace("--", " ").replace("~", " ")
    abstract_words = len(re.findall(r"[A-Za-z0-9][\w.,;:()%/-]*", plain))
    check("abstract <= 250 words (CMPB guide)", abstract_words <= 250,
          f"{abstract_words} words")
    keywords = [k.strip() for k in
                tex.split("\\begin{keyword}")[1].split("\\end{keyword}")[0]
                .replace("\n", " ").split("\\sep")]
    check("3-6 keywords", 3 <= len(keywords) <= 6, f"{len(keywords)}: {', '.join(keywords)}")

    highlights = [line for line in
                  (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
                  if line.strip() and line.strip() != "Highlights"]
    check("highlights: 3-5 bullets, each <= 85 characters",
          3 <= len(highlights) <= 5 and max(len(h) for h in highlights) <= 85,
          f"{len(highlights)} bullets, longest {max(len(h) for h in highlights)}")

    # --- required declarations ---------------------------------------------
    required = {
        "data availability": "Data and code availability",
        "ethics statement": "Ethics statement",
        "CRediT": "CRediT authorship contribution statement",
        "funding": "Funding",
        "competing interest": "Declaration of competing interest",
        "generative AI declaration": "Declaration of generative AI",
    }
    for label, needle in required.items():
        check(f"declaration present: {label}", needle in tex)
    check("generative AI declaration placed before the bibliography",
          tex.index("Declaration of generative AI") < tex.index("\\bibliographystyle"))

    # --- figures and tables -------------------------------------------------
    figures = re.findall(r"\\label\{(fig:[^}]+)\}", tex)
    tables = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    refs = set(re.findall(r"\\ref\{((?:fig|tab):[^}]+)\}", tex))
    missing = [item for item in figures + tables if item not in refs]
    check("every figure/table label is cited in the text", not missing, ",".join(missing))
    figure_refs = re.findall(r"\\ref\{(fig:[^}]+)\}", tex)
    check("figure/table counts", len(figures) >= 1 and len(tables) >= 1,
          f"{len(figures)} figures, {len(tables)} tables, "
          f"{len(figure_refs)} in-text figure references")

    # --- leftovers ----------------------------------------------------------
    leftovers = [token for token in ("TODO", "TBD", "XXX", "FIXME", "???", "PLACEHOLDER")
                 if re.search(re.escape(token), tex)]
    check("no placeholders left in the manuscript", not leftovers, ",".join(leftovers))
    check("line numbers enabled for review",
          "\\reviewmodetrue" in tex and "\\linenumbers" in tex)

    # --- references ---------------------------------------------------------
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),", bib)
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(name.strip() for name in group.split(","))
    uncited = sorted(set(entries) - cited)
    undefined = sorted(cited - set(entries))
    check("every bibliography entry is cited", not uncited, ",".join(uncited))
    check("every citation resolves to an entry", not undefined, ",".join(undefined))
    check("reference count", len(entries) >= 10, f"{len(entries)} entries, "
          f"{len(cited)} cited")
    log = (PAPER / "references_log.md").read_text(encoding="utf-8")
    check("reference audit trail covers every entry",
          all(name in log for name in entries))

    failures = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failures:
        print(f"{len(failures)} ITEM(S) NEED ATTENTION: {', '.join(failures)}")
        return 1
    print(f"ALL {len(CHECKS)} STRUCTURAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
