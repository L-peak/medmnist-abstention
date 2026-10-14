"""Local test for src.data.ensure_dataset_file: download, restart, retry, MD5 check.

Serves a file over localhost so the retry/resume logic can be exercised without
touching Zenodo. Python's http.server does not honour Range requests, so this also
verifies the "server ignores Range -> restart from scratch" branch.
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ensure_dataset_file, dataset_source, file_md5  # noqa: E402


def serve(directory: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Quiet(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        payload = bytes(range(256)) * 8192          # 2 MB, deterministic
        served = tmp_path / "served.npz"
        served.write_bytes(payload)
        digest = file_md5(served)
        httpd, base = serve(tmp_path)
        target_root = tmp_path / "client"
        try:
            # 1. fresh download
            got = ensure_dataset_file("demo", 28, root=target_root,
                                      url=f"{base}/served.npz", md5=digest, verbose=False)
            assert got.exists() and file_md5(got) == digest
            print("[ok] fresh download + md5 verified")

            # 2. already present -> no re-download (mtime unchanged)
            before = got.stat().st_mtime_ns
            ensure_dataset_file("demo", 28, root=target_root,
                                url=f"{base}/served.npz", md5=digest, verbose=False)
            assert got.stat().st_mtime_ns == before, "file was re-downloaded"
            print("[ok] existing valid file is reused")

            # 3. truncated file -> re-download (server ignores Range, so restart)
            with open(got, "r+b") as handle:
                handle.truncate(len(payload) // 3)
            ensure_dataset_file("demo", 28, root=target_root,
                                url=f"{base}/served.npz", md5=digest, verbose=False)
            assert file_md5(got) == digest
            print("[ok] truncated file is repaired")

            # 4. wrong md5 -> retries and then fails loudly
            try:
                ensure_dataset_file("demo", 28, root=target_root,
                                    url=f"{base}/served.npz", md5="0" * 32, retries=2, verbose=False)
                raise AssertionError("expected RuntimeError for md5 mismatch")
            except RuntimeError as exc:
                assert "could not download" in str(exc)
                print("[ok] md5 mismatch fails loudly instead of silently continuing")
        finally:
            httpd.shutdown()

    # 5. real dataset already on disk -> instant return
    real = ensure_dataset_file("breastmnist", 28, verbose=False)
    print(f"[ok] real dataset fast path: {real.name} ({real.stat().st_size / 1e6:.1f} MB)")
    for flag, size in (("pathmnist", 128),):
        url, md5 = dataset_source(flag, size)
        print(f"[info] {flag}_{size} source = {url.split('/')[-1]}  md5={md5}")

    print("\nDOWNLOAD TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
