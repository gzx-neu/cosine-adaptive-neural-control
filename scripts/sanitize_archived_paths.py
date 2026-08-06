"""Replace machine-specific archived paths with repository-relative metadata."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".log", ".err", ".tex"}

# JSON text contains doubled backslashes; CSV/log text usually contains one.
PREFIX_PATTERNS = (
    re.compile(r"C:\\\\Users\\\\[^\\\"\r\n]+\\\\Desktop\\\\.*?\\\\(?=(?:kkt_collocation|offline_safe_control|论文写作)\\\\)"),
    re.compile(r"C:\\Users\\[^\\\"\r\n]+\\Desktop\\.*?\\(?=(?:kkt_collocation|offline_safe_control|论文写作)\\)"),
)
PYTHON_PATTERNS = (
    re.compile(r"[A-Za-z]:\\\\[^\"\r\n]*?python\.exe", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^\"\r\n]*?python\.exe", re.IGNORECASE),
)
PRIVATE_PATH_FALLBACKS = (
    re.compile(r"C:\\\\Users\\\\[^\\\"\r\n]+\\\\[^\"\r\n]*", re.IGNORECASE),
    re.compile(r"C:\\Users\\[^\\\"\r\n]+\\[^\"\r\n]*", re.IGNORECASE),
)


def main() -> None:
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            original = path.read_text(encoding="utf-8-sig")
        updated = original
        for pattern in PREFIX_PATTERNS:
            updated = pattern.sub("", updated)
        for pattern in PYTHON_PATTERNS:
            updated = pattern.sub("python", updated)
        for pattern in PRIVATE_PATH_FALLBACKS:
            updated = pattern.sub("<archived-local-path>", updated)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    remaining = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"C:\\\\Users\\\\[^\\\"\r\n]+", text) or re.search(r"C:\\Users\\[^\\\"\r\n]+", text):
                remaining.append(path.relative_to(ROOT).as_posix())
    if remaining:
        raise RuntimeError("Private archived paths remain:\n" + "\n".join(remaining[:20]))
    print(f"sanitized machine-specific metadata in {changed} files")


if __name__ == "__main__":
    main()
