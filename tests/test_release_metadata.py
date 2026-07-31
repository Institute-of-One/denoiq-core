"""The archived release: one version everywhere, and no DOI invented before it exists.

The manuscript cites a Zenodo record of this repository. Two things can go wrong at submission
time and neither is visible in a rendered PDF: the archived version can disagree with the code
that produced the numbers, and a placeholder DOI can survive into the file that reviewers read.
Both are checked here rather than in a checklist, because a checklist is not run by CI.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
sys.path.insert(0, str(PAPER))

import build_manuscript  # noqa: E402

RELEASE = json.loads((PAPER / "release.json").read_text(encoding="utf-8"))


def test_every_file_declares_the_same_version():
    """A Zenodo record archives one version; five files name it."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    init = (REPO / "denoiq_core" / "__init__.py").read_text(encoding="utf-8")
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))

    version = RELEASE["version"]
    assert re.search(rf'^version = "{re.escape(version)}"$', pyproject, re.M)
    assert re.search(rf'^__version__ = "{re.escape(version)}"$', init, re.M)
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)
    assert zenodo["version"] == version
    assert RELEASE["tag"] == f"v{version}"


def test_the_draft_states_the_doi_is_pending_without_inventing_one():
    """Before the release exists the text says so, in words, with no token to mistake."""
    statement = build_manuscript.release_value(
        "archive_statement", {"version": "0.1.0", "archive": "Zenodo"}, submission=False
    )
    assert "0.1.0" in statement
    assert "inserted at submission" in statement
    for token in ("PENDING", "TODO", "XXX", "10.5281"):
        assert token not in statement


def test_a_submission_cannot_be_built_until_the_doi_is_minted():
    """The rule that keeps a placeholder out of a submitted PDF."""
    with pytest.raises(build_manuscript.ResolutionError, match="version_doi"):
        build_manuscript.release_value("archive_statement", {"version": "0.1.0"}, submission=True)


def test_a_minted_doi_is_printed_in_the_manuscript():
    """And once it exists, it is the DOI that reaches the page — in both modes."""
    release = {"version": "0.1.0", "archive": "Zenodo", "version_doi": "10.5281/zenodo.1234567"}
    for submission in (False, True):
        statement = build_manuscript.release_value(
            "archive_statement", release, submission=submission
        )
        assert "doi:10.5281/zenodo.1234567" in statement
        assert "version 0.1.0" in statement


def test_the_manuscript_takes_its_repository_url_from_the_release_file():
    """The URL is written once. A second copy in the prose could drift from the archive."""
    source = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    assert "[[release:repository]]" in source
    rendered = build_manuscript.render(source, REPO / "results")
    assert RELEASE["repository"] in rendered


def test_the_release_file_carries_no_invented_identifier():
    """Nothing may fill in the DOI except the archive that mints it."""
    for field in ("version_doi", "concept_doi"):
        value = RELEASE[field]
        assert value is None or re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", value), value
