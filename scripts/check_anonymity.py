"""Scan the repository for anything that would break the anonymity of the review copy.

Files scanned: every text file that git would track, found without calling git: the
whole repository except .git, .venv, results/, caches, paths matched by .gitignore,
binary files, and the term list files themselves. --extra adds files or folders
outside that set, including PDFs (text and metadata read from the raw and the
decompressed streams) and .tex files.

Checks:
  terms     each line of the terms file (default scripts/anonymity_terms.local.txt,
            lines starting with # are comments), case insensitive; a space or
            underscore in a term also matches no separator, an underscore, a space,
            a hyphen or a dot, so "first last" finds "FirstLast" and "first_last"
  email     any email address
  url       any URL on GitHub, LinkedIn or ResearchGate
  path      any absolute path that names a user folder (Windows Users folder,
            /home/<name>, /Users/<name>)

Prints file, line number and the match. Exit code 0 when nothing is found, 1 when
something is found.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\check_anonymity.py
    .venv\\Scripts\\python.exe scripts\\check_anonymity.py --terms scripts\\anonymity_terms.example.txt
    .venv\\Scripts\\python.exe scripts\\check_anonymity.py --extra paper\\main.pdf paper\\tex
"""

import argparse
import os
import re
import sys
import zlib
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
TERMS_PATH = ROOT / "scripts" / "anonymity_terms.local.txt"
TERM_LISTS = ("scripts/anonymity_terms.local.txt", "scripts/anonymity_terms.example.txt")
ALWAYS_SKIPPED_DIRS = {".git", ".venv", "venv", "results", "__pycache__", ".pytest_cache", ".mypy_cache",
                       ".ipynb_checkpoints", "node_modules"}
BINARY_SUFFIXES = {".db", ".sqlite", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc", ".zip", ".gz", ".xlsx",
                   ".docx", ".pptx", ".ico", ".woff", ".woff2", ".ttf", ".otf"}
EXTRA_SUFFIXES = {".pdf", ".tex", ".bib", ".sty", ".cls", ".md", ".txt", ".csv", ".json", ".py", ".yaml"}

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
URL = re.compile(r"(?:https?://)?(?:[\w\-]+\.)*(?:github|linkedin)\.com\S*|(?:https?://)?(?:[\w\-]+\.)*researchgate\.net\S*",
                 re.IGNORECASE)
SEP = r"(?:\\\\|\\|/)+"
USER_PATH = re.compile(
    rf"\b[A-Za-z]:{SEP}Users{SEP}([^\\/\s\"'`,;:)]+)|(?<![\w.])/home/([^/\s\"'`,;:)]+)|(?<![\w.:])/Users/([^/\s\"'`,;:)]+)",
    re.IGNORECASE,
)
# Path user parts that are placeholders, not real user names.
PLACEHOLDER_NAMES = {"user", "username", "name", "you", "yourname", "public", "default", "all users"}


def read_terms(path):
    terms = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def term_regex(term):
    parts = [re.escape(part) for part in re.split(r"[\s_]+", term) if part]
    return re.compile(r"[\s_\-.]*".join(parts), re.IGNORECASE)


def gitignore_patterns(root):
    path = Path(root) / ".gitignore"
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            patterns.append(line)
    return patterns


def is_ignored(rel, is_dir, patterns):
    name = PurePosixPath(rel).name
    for pattern in patterns:
        dir_only = pattern.endswith("/")
        body = pattern.strip("/")
        if dir_only and not is_dir:
            continue
        anchored = "/" in body
        if fnmatch(rel, body) if anchored else fnmatch(name, body):
            return True
    return False


def repo_files(root, skip=()):
    """Text file candidates under root, the way git would see them, without running git."""
    root = Path(root)
    skip = {Path(path).resolve() for path in skip}
    patterns = gitignore_patterns(root)
    for current, dirs, files in os.walk(root):
        rel_dir = Path(current).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirs[:] = sorted(
            d for d in dirs
            if d not in ALWAYS_SKIPPED_DIRS and not is_ignored(rel_dir + d, True, patterns)
        )
        for name in sorted(files):
            rel = rel_dir + name
            if rel in TERM_LISTS or is_ignored(rel, False, patterns) or (Path(current) / name).resolve() in skip:
                continue
            if Path(name).suffix.lower() in BINARY_SUFFIXES:
                continue
            yield Path(current) / name


def extra_files(paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for found in sorted(path.rglob("*")):
                if found.is_file() and found.suffix.lower() in EXTRA_SUFFIXES:
                    yield found
        elif path.is_file():
            yield path
        else:
            print(f"warning: {path} does not exist")


def pdf_text(data):
    """Text of a PDF from its raw bytes and every stream zlib can inflate.

    Strings split by kerning inside TJ arrays are joined; text in fonts with custom
    encodings may stay unreadable, so a clean PDF result is a hint, not a proof.
    """
    chunks = [data]
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
        try:
            chunks.append(zlib.decompress(match.group(1)))
        except zlib.error:
            continue
    text = b"\n".join(chunks).decode("latin-1")
    return re.sub(r"\)\s*-?\d+(?:\.\d+)?\s*\(", "", text)


def file_text(path):
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() == ".pdf":
        return pdf_text(data)
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def placeholder_user(name):
    """True for a part like <name>, %USERNAME% or ..., which no real user folder starts with."""
    lowered = name.strip().lower()
    return not lowered[:1].isalnum() or lowered in PLACEHOLDER_NAMES


def scan_text(text, terms):
    """(line number, kind, match) for every finding in the text."""
    findings = []
    compiled = [(term, term_regex(term)) for term in terms]
    for number, line in enumerate(text.splitlines(), start=1):
        for term, regex in compiled:
            if regex.search(line):
                findings.append((number, "term", term))
        for match in EMAIL.finditer(line):
            findings.append((number, "email", match.group(0)))
        for match in URL.finditer(line):
            findings.append((number, "url", match.group(0)))
        for match in USER_PATH.finditer(line):
            user = next(group for group in match.groups() if group is not None)
            if not placeholder_user(user):
                findings.append((number, "path", match.group(0)))
    return findings


def display(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--terms", type=Path, default=TERMS_PATH, help="terms file, one term per line")
    parser.add_argument("--extra", nargs="+", default=[], help="extra files or folders, PDFs and .tex included")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.terms.exists():
        terms = read_terms(args.terms)
        print(f"terms: {len(terms)} from {display(args.terms, args.root)}")
    else:
        terms = []
        print(f"terms: none ({display(args.terms, args.root)} not found; copy"
              " scripts/anonymity_terms.example.txt to it and fill in the real terms). Generic checks only.")

    files = list(repo_files(args.root, skip=[args.terms])) + list(extra_files(args.extra))
    scanned, skipped, counts, total = 0, [], {}, 0
    for path in files:
        text = file_text(path)
        if text is None:
            skipped.append(display(path, args.root))
            continue
        scanned += 1
        for number, kind, match in scan_text(text, terms):
            counts[kind] = counts.get(kind, 0) + 1
            total += 1
            shown = match if len(match) <= 120 else match[:117] + "..."
            print(f"{display(path, args.root)}:{number}: {kind}: {shown}")
    print()
    print(f"scanned {scanned} text file(s); skipped {len(skipped)} binary file(s)"
          + (f": {', '.join(skipped)}" if skipped else ""))
    summary = ", ".join(f"{kind} {counts[kind]}" for kind in ("term", "email", "url", "path") if kind in counts)
    print(f"findings: {total}" + (f" ({summary})" if summary else ""))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
