"""Anonymity scanner tests on a temporary repository with planted findings.

Planted values are assembled at run time so this file itself never matches a check.
"""

import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_anonymity as scan

AT = "@"
EMAIL = "jane.placeholder" + AT + "example" + ".org"
GITHUB = "https://" + "git" + "hub.com/someone/repo"
WIN_PATH = "C:" + "\\" + "Users" + "\\" + "jdoe" + "\\" + "project"
HOME_PATH = "/ho" + "me/jdoe/project"


def make_repo(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored_dir/\n*.log\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "anonymity_terms.local.txt").write_text("Jane Placeholder\n", encoding="utf-8")
    (tmp_path / "terms.txt").write_text("# comment\nJane Placeholder\nAcme Lab\n", encoding="utf-8")
    (tmp_path / "a.md").write_text(
        f"line one\nby JanePlaceholder and acme_lab\ncontact {EMAIL}\ncode at {GITHUB}\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(f"PATH = r'{WIN_PATH}'\nOTHER = '{HOME_PATH}'\nSAFE = '/home/<name>/x'\n",
                                   encoding="utf-8")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "c.md").write_text("Jane Placeholder\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("Jane Placeholder\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "d.md").write_text("Jane Placeholder\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\0\0Jane Placeholder")
    return tmp_path


def test_term_regex_matches_spacing_variants():
    regex = scan.term_regex("Jane Placeholder")
    for text in ("jane placeholder", "JanePlaceholder", "jane_placeholder", "Jane-Placeholder", "jane.placeholder"):
        assert regex.search(text)
    assert not regex.search("Jane P.")


def test_scan_finds_terms_emails_urls_and_user_paths(tmp_path, capsys):
    root = make_repo(tmp_path)
    code = scan.main(["--root", str(root), "--terms", str(root / "terms.txt")])
    out = capsys.readouterr().out
    assert code == 1
    assert "a.md:2: term: Jane Placeholder" in out
    assert "a.md:2: term: Acme Lab" in out
    assert "a.md:3: term: Jane Placeholder" in out  # the email holds the name as jane.placeholder
    assert f"a.md:3: email: {EMAIL}" in out
    assert "a.md:4: url: " in out
    assert "b.py:1: path: " in out and "b.py:2: path: " in out
    assert "b.py:3:" not in out
    for skipped in ("ignored_dir", "run.log", "results/", "anonymity_terms.local.txt", "terms.txt:"):
        assert skipped not in out.replace("skipped", "")
    assert "blob.bin" in out.split("skipped")[1]


def test_clean_repo_exits_zero_and_missing_terms_file_is_reported(tmp_path, capsys):
    (tmp_path / "clean.md").write_text("nothing to see\n", encoding="utf-8")
    assert scan.main(["--root", str(tmp_path), "--terms", str(tmp_path / "missing.txt")]) == 0
    assert "Generic checks only" in capsys.readouterr().out


def test_extra_pdf_text_and_metadata_are_scanned(tmp_path, capsys):
    stream = zlib.compress(b"BT [(Jane Place)-20(holder)] TJ ET")
    pdf = (b"%PDF-1.4\n1 0 obj << /Author (Acme Lab) >> endobj\n2 0 obj << /Filter /FlateDecode >>\nstream\n"
           + stream + b"\nendstream\nendobj\n%%EOF\n")
    (tmp_path / "paper.pdf").write_bytes(pdf)
    (tmp_path / "paper.tex").write_text("\\author{Jane Placeholder}\n", encoding="utf-8")
    (tmp_path / "terms.txt").write_text("Jane Placeholder\nAcme Lab\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    code = scan.main(["--root", str(repo), "--terms", str(tmp_path / "terms.txt"),
                      "--extra", str(tmp_path / "paper.pdf"), str(tmp_path / "paper.tex")])
    out = capsys.readouterr().out
    assert code == 1
    assert "paper.pdf" in out and "term: Acme Lab" in out and "term: Jane Placeholder" in out
    assert "paper.tex:1: term: Jane Placeholder" in out
