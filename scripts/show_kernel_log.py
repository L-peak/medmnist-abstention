"""Show the useful parts of a Kaggle kernel log (structured JSON rows).

Kaggle returns one JSON object per log event: {"stream_name", "time", "data"}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    events = []
    for raw in text.splitlines():
        row = raw.strip().rstrip(",")
        prefix = '{"stream_name"'
        start = row.find(prefix)
        if start == -1:
            continue
        try:
            events.append(json.loads(row[start:]))
        except Exception:
            continue
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="path to the kernel .log file")
    parser.add_argument("--errors-only", action="store_true")
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    events = load_events(Path(args.log))
    print(f"parsed {len(events)} log events")
    for event in events:
        data = str(event.get("data", ""))
        interesting = ("[error]" in data or "Traceback" in data or "Error" in data
                       or "done:" in data or "zip:" in data)
        if args.errors_only and not interesting:
            continue
        if not interesting:
            continue
        print("-" * 70)
        print(data.strip()[: args.max_chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
