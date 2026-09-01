#!/usr/bin/env python3
"""Promote the verified historical Zchezz WASM template to engine/build/.

The historical template is tracked in v401/v314. On Windows, Git may check
those files out with CRLF line endings, while the Git blob itself is stored
with LF. Verification therefore normalizes CRLF/CR to LF before computing
the expected Git blob SHA.

The promoted file preserves the source file's working-tree bytes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "engine" / "build" / "zchezz_wasm.html"

# Canonical Git blob SHA of the LF-normalized historical template.
EXPECTED_GIT_BLOB_SHA1 = "41302a7f3df856c1e41154c5546a869463897cc4"

SOURCES = (
    ROOT / "engine" / "c" / "zchezz_v401" / "zchezz_wasm.html",
    ROOT / "engine" / "c" / "zchezz_v314" / "zchezz_wasm.html",
)


def normalize_text_bytes(data: bytes) -> bytes:
    """Normalize text line endings to Git-canonical LF form."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def canonical_blob_sha1(data: bytes) -> str:
    return git_blob_sha1(normalize_text_bytes(data))


def verified_source() -> tuple[Path, bytes]:
    problems: list[str] = []

    for source in SOURCES:
        if not source.is_file():
            problems.append(f"missing {source.relative_to(ROOT)}")
            continue

        data = source.read_bytes()
        raw_digest = git_blob_sha1(data)
        canonical_digest = canonical_blob_sha1(data)

        if canonical_digest == EXPECTED_GIT_BLOB_SHA1:
            if raw_digest != canonical_digest:
                print(
                    "INFO: Windows line-ending conversion detected for "
                    f"{source.relative_to(ROOT)}"
                )
                print(f"      working-tree blob SHA: {raw_digest}")
                print(f"      LF-normalized blob SHA: {canonical_digest}")
            return source, data

        problems.append(
            f"{source.relative_to(ROOT)} has canonical Git blob SHA "
            f"{canonical_digest}; expected {EXPECTED_GIT_BLOB_SHA1} "
            f"(raw working-tree SHA {raw_digest})"
        )

    raise RuntimeError(
        "No verified template source.\n  " + "\n  ".join(problems)
    )


def verify_target() -> int:
    if not TARGET.is_file():
        print(f"ERROR: shared WASM template missing: {TARGET.relative_to(ROOT)}")
        return 1

    data = TARGET.read_bytes()
    raw_digest = git_blob_sha1(data)
    canonical_digest = canonical_blob_sha1(data)

    if canonical_digest != EXPECTED_GIT_BLOB_SHA1:
        print(
            "ERROR: shared WASM template differs from the verified historical "
            f"template.\n  canonical SHA: {canonical_digest}\n"
            f"  expected SHA:  {EXPECTED_GIT_BLOB_SHA1}\n"
            f"  raw SHA:       {raw_digest}"
        )
        return 1

    print(
        f"PASS: {TARGET.relative_to(ROOT)} verified "
        f"(canonical Git blob SHA {canonical_digest})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the promoted shared template instead of creating it",
    )
    args = parser.parse_args()

    if args.check:
        return verify_target()

    try:
        source, data = verified_source()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if TARGET.is_file():
        existing = canonical_blob_sha1(TARGET.read_bytes())
        if existing == EXPECTED_GIT_BLOB_SHA1:
            print(f"PASS: shared template already exists: {TARGET.relative_to(ROOT)}")
            return 0

        print(
            f"ERROR: refusing to overwrite modified {TARGET.relative_to(ROOT)} "
            f"(canonical Git blob SHA {existing})"
        )
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)

    print(
        f"PASS: promoted {source.relative_to(ROOT)} -> "
        f"{TARGET.relative_to(ROOT)}"
    )
    print(f"Canonical Git blob SHA: {EXPECTED_GIT_BLOB_SHA1}")
    print("IMPORTANT: commit engine/build/zchezz_wasm.html.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
