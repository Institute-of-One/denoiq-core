"""Every number in the manuscript is re-derived from ``results/``.

The failure this catches is mundane and common: a result changes, the code and the figures
are regenerated, and a number in the text — typed by hand three revisions ago — quietly stops
being true. Here the text cannot contain a typed number at all. Each one is a marker naming a
results file and a path into it, the build resolves them, and this test rebuilds and compares.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
RESULTS = REPO / "results"
SOURCE = PAPER / "manuscript.md"
BUILT = PAPER / "build" / "manuscript.md"

sys.path.insert(0, str(PAPER))
build_manuscript = pytest.importorskip("build_manuscript")

REQUIRED_RESULTS = (
    "summary.json",
    "dose_sweep.json",
    "closed_form.json",
    "task_gains.json",
    "redlamp_atlas.json",
    "taskbench_cnn.json",
    "redlamp_console.json",
)

#: Lines that may carry digits without citing a result: headings (section numbers), the
#: author block (ORCID), and ordered lists. Everything else must use a marker — or, for a
#: constant of the method rather than a measurement, a code span, which is stripped below.
EXEMPT_LINE = re.compile(r"^\s*#|^\s*\d+\.\s|ORCID")
NUMBER = re.compile(r"\d+\.\d+|\b\d{2,}\b")


def _have_results() -> bool:
    return all((RESULTS / name).exists() for name in REQUIRED_RESULTS)


requires_results = pytest.mark.skipif(
    not _have_results(),
    reason=(
        "results/ has not been generated in this checkout: run "
        '`python -c "from denoiq_core.experiment import run_all; run_all()"`'
    ),
)


def test_the_manuscript_source_exists():
    assert SOURCE.exists(), "paper/manuscript.md is the content of record and must be present"


@requires_results
def test_every_marker_resolves():
    """A citation that names nothing in results/ is a fabricated number."""
    text = SOURCE.read_text(encoding="utf-8")
    build_manuscript.render(text, RESULTS)  # raises ResolutionError if any marker fails


@requires_results
def test_the_built_manuscript_is_up_to_date():
    """The committed build must equal a fresh render: no hand edits, no stale numbers."""
    if not BUILT.exists():
        pytest.skip("paper/build/manuscript.md has not been built yet")
    fresh = build_manuscript.render(SOURCE.read_text(encoding="utf-8"), RESULTS)
    assert BUILT.read_text(encoding="utf-8") == fresh, (
        "paper/build/manuscript.md is out of date — rebuild it with "
        "`python paper/build_manuscript.py`"
    )


def test_no_numbers_are_typed_into_the_prose():
    """Numbers must come from markers, not from a keyboard.

    Code spans, fenced blocks, HTML comments, links and the reference list are exempt: they
    are notation and bibliography, not results.
    """
    text = SOURCE.read_text(encoding="utf-8")
    text = text.split("## References")[0]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\[\[results:[^\]]*\]?[^\]]*\]\]", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    # Cross-references and citation markers are pointers, not measurements.
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(
        r"\b(?:Sections?|Figures?|Tables?)\s*\d+(?:\.\d+)?"
        r"(?:\s*(?:and|,|–|-)\s*\d+(?:\.\d+)?)*",
        "",
        text,
    )

    offenders = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if EXEMPT_LINE.search(line):
            continue
        for match in NUMBER.finditer(line):
            offenders.append(f"line {lineno}: {match.group(0)!r} in {line.strip()[:80]!r}")
    assert not offenders, "typed numbers in the manuscript prose:\n  " + "\n  ".join(offenders)


@requires_results
def test_the_headline_claim_is_the_one_in_the_results():
    """The abstract says the ceiling held everywhere. It must actually have held."""
    import json

    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["bound_holds_everywhere"] is True
    assert summary["n_bound_violations"] == 0
    assert summary["n_conditions"] > 0
    # ...and the estimator was not so weak that holding was inevitable.
    assert summary["min_estimator_d_prime_ratio"] > 0.5


@requires_results
def test_results_files_record_their_provenance():
    """Every results file says which versions produced it."""
    import json

    for name in REQUIRED_RESULTS:
        payload = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        provenance = payload.get("provenance")
        assert provenance, f"{name} has no provenance block"
        assert provenance["denoiq_core"]
        assert provenance["taskiq_core"]


@requires_results
def test_the_manuscript_does_not_quote_a_quick_run():
    """A quick run must not be what the manuscript is built from.

    ``run_all(quick=True)`` writes small, fast sweeps for smoke tests; they are not
    publishable numbers.
    """
    import json

    for name in REQUIRED_RESULTS:
        payload = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        assert payload["provenance"].get("quick") is not True, (
            f"{name} came from a quick run; regenerate with run_all()"
        )


def test_the_version_is_the_same_in_every_place_it_appears():
    """`__version__`, pyproject and CITATION.cff must agree.

    A release is archived by version; three files claiming three versions is how a DOI ends
    up pointing at code that is not the code the manuscript used.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    import denoiq_core

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    packaged = pyproject["project"]["version"]
    assert packaged == denoiq_core.__version__

    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {packaged}" in citation, "CITATION.cff version is out of step"

    import json

    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["version"] == packaged, ".zenodo.json version is out of step"


def test_the_affiliation_policy_holds_in_the_manuscript():
    """Only one affiliation may appear, and two institutions may appear nowhere.

    CI checks this across every tracked file; here it is checked where it matters most, so
    that a local run catches it before a push does.
    """
    text = SOURCE.read_text(encoding="utf-8")
    assert "Institute of One, LISIT Co., Ltd., Tokyo, Japan" in text
    for forbidden in ("National Cancer Center", "Tohoku University"):
        assert forbidden not in text


# --------------------------------------------------------------------------------------
# terminology: the claims the revision pinned down must stay pinned down
# --------------------------------------------------------------------------------------

#: Phrases that overstate the result. Each one was in an earlier draft; each is wrong for a
#: stated reason, so a regression is a claim regression rather than a style lapse.
BANNED_IN_MANUSCRIPT = (
    # The bound is about the likelihood-ratio ideal observer; what we measure on processed
    # images is an estimated linear observer, and calling it ideal overstates the measurement.
    "processed ideal observer",
    "processed ideal-observer",
    "ideal observer after processing",
    "ideal-observer AUC after processing",
    # The floor is a prespecified requirement, not a zero-information boundary.
    "plausibility, not information",
    "task becomes impossible",
    "no amount of processing restores",
    "the information is not there",
    "was never acquired",
    # d' = 1 is an adequacy threshold (AUC ~ 0.76), not chance (AUC 0.5).
    "chance-level",
    "at chance:",
    # NPWE is a stylised surrogate; no human observer study was run.
    "the human surrogate",
)

#: Statements the paper must actually make, so the operational reading cannot be lost in a
#: later edit.
REQUIRED_IN_MANUSCRIPT = (
    "held-out prewhitening linear observer",
    "likelihood-ratio ideal observer",
    "not a boundary of zero information",
    "prespecified",
    "stylized surrogate",
    "setting-informed (oracle) parameterization",
)

#: Sources that write human-visible labels: figure axes, console text, table headers.
LABEL_SOURCES = (
    REPO / "denoiq_core" / "figures.py",
    REPO / "denoiq_core" / "redlamp.py",
    REPO / "paper" / "redlamp_console.py",
    REPO / "paper" / "build_pdf.py",
)


def test_the_manuscript_does_not_overstate_the_result():
    text = SOURCE.read_text(encoding="utf-8")
    body = text.split("-->", 1)[1]  # the header comment documents the banned phrases
    offenders = [phrase for phrase in BANNED_IN_MANUSCRIPT if phrase.lower() in body.lower()]
    assert not offenders, f"overstated phrasing is back in the manuscript: {offenders}"


def test_the_manuscript_states_the_things_it_must_state():
    text = SOURCE.read_text(encoding="utf-8")
    missing = [phrase for phrase in REQUIRED_IN_MANUSCRIPT if phrase not in text]
    assert not missing, f"the manuscript no longer states: {missing}"


def test_figure_and_console_labels_name_the_observer_that_was_measured():
    """No axis, legend, table header or console string may call the estimate an ideal observer."""
    offenders = []
    for path in LABEL_SOURCES:
        lowered = path.read_text(encoding="utf-8").lower()
        for phrase in (
            "processed ideal",
            "ideal-observer auc after processing",
            "chance-level",
            "the human surrogate",
            "plausibility, not information",
        ):
            if phrase in lowered:
                offenders.append(f"{path.name}: {phrase!r}")
    assert not offenders, "stale observer/floor labels in generated output:\n  " + "\n  ".join(
        offenders
    )


# --------------------------------------------------------------------------------------
# the numbers the manuscript quotes, recomputed from the sweep files
# --------------------------------------------------------------------------------------


def _all_rows() -> list[dict]:
    import json

    rows = []
    for name in ("dose_sweep", "texture_sweep", "signal_sweep", "redlamp_demo"):
        rows.extend(json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))["rows"])
    return rows


def _summary() -> dict:
    import json

    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


@requires_results
def test_the_condition_count_is_recomputable_and_excludes_the_cnn():
    """76 = 21 unprocessed + 55 processed, and the CNN is not among them."""
    rows = _all_rows()
    summary = _summary()
    processed = [row for row in rows if row["method"] != "none"]
    assert summary["n_conditions"] == len(rows)
    assert summary["n_processed_conditions"] == len(processed)
    assert summary["n_unprocessed_conditions"] == len(rows) - len(processed)
    assert summary["n_conditions"] == (
        summary["n_processed_conditions"] + summary["n_unprocessed_conditions"]
    )
    # The primary analysis is classical-only, so it reproduces without torch.
    assert summary["cnn_in_primary_analysis"] is False
    assert not any(row["method"] == "cnn" for row in rows)
    assert len(summary["denoisers_in_primary_analysis"]) == 3
    assert sum(summary["arms_per_sweep"].values()) == summary["n_conditions"]


@requires_results
def test_the_headline_statistics_are_recomputable():
    """Every summary number the abstract and Section 3.3 quote, recomputed from the rows."""
    import statistics

    rows = _all_rows()
    summary = _summary()
    processed = [row for row in rows if row["method"] != "none"]

    assert summary["max_excess_over_ceiling"] == pytest.approx(max(r["excess"] for r in rows))
    assert summary["mean_excess_over_ceiling"] == pytest.approx(
        statistics.fmean(r["excess"] for r in rows)
    )
    assert summary["mean_excess_over_ceiling_processed_only"] == pytest.approx(
        statistics.fmean(r["excess"] for r in processed)
    )
    assert summary["n_bound_violations"] == sum(1 for r in rows if not r["bound_ok"])
    assert summary["min_estimator_d_prime_ratio"] == pytest.approx(
        min(r["estimator_d_prime_ratio"] for r in rows)
    )
    assert summary["n_auc_saturated_arms"] == sum(1 for r in rows if r["saturated"])
    for level, count in summary["levels"].items():
        assert count == sum(1 for r in rows if r["level"] == level)
    assert sum(summary["levels"].values()) == summary["n_conditions"]

    # The margin quoted next to the largest excess belongs to that same arm.
    worst = max(rows, key=lambda r: r["excess"])
    assert summary["margin_at_max_excess"] == pytest.approx(worst["margin"])
    assert summary["max_excess_over_ceiling"] <= summary["margin_at_max_excess"]
    assert summary["max_excess_is_auc_saturated"] == worst["saturated"]


@requires_results
def test_the_npwe_gain_is_the_largest_one_in_the_study():
    rows = _all_rows()
    summary = _summary()
    from denoiq_core.experiment import _matching_raw

    gains = []
    for row in rows:
        if row["method"] == "none":
            continue
        raw = _matching_raw(rows, row)
        if raw is not None:
            gains.append(row["d_prime_npwe"] - raw["d_prime_npwe"])
    assert summary["max_npwe_gain_from_denoising"] == pytest.approx(max(gains))
    assert max(gains) > 0.0


@requires_results
def test_table_1_and_table_2_agree_with_their_sources():
    import json

    closed = json.loads((RESULTS / "closed_form.json").read_text(encoding="utf-8"))
    assert closed["max_relative_error"] == pytest.approx(
        max(row["relative_error"] for row in closed["rows"])
    )
    assert closed["max_relative_error"] < 0.01  # the prespecified acceptance criterion

    gains = json.loads((RESULTS / "task_gains.json").read_text(encoding="utf-8"))
    dose_rows = [row for row in gains["rows"] if row["sweep"] == "dose"]
    for name, entry in gains["by_denoiser"].items():
        subset = [row for row in dose_rows if row["denoiser"] == name]
        assert subset, f"no dose-sweep rows for {name}"
        for observer in ("ideal", "cho", "npwe"):
            key = f"mean_delta_d_prime_{observer}"
            recomputed = sum(row[f"delta_d_prime_{observer}"] for row in subset) / len(subset)
            assert entry[key] == pytest.approx(recomputed)
        # Table 2's claim: the non-prewhitening observer gains, the prewhitening one does not.
        assert entry["mean_delta_d_prime_npwe"] > 0.0
        assert entry["mean_delta_d_prime_ideal"] <= 0.0


@requires_results
def test_the_floor_is_linear_in_kv_as_the_model_implies():
    """`mAs_floor ∝ kV` for this model; Section 3.6 quotes the two endpoints."""
    import json

    atlas = json.loads((RESULTS / "redlamp_atlas.json").read_text(encoding="utf-8"))["atlas"]
    kv = atlas["kv"]
    floor = atlas["floor_mas"]
    ratios = [f / k for f, k in zip(floor, kv, strict=True)]
    # Not exact to machine precision: the disk is area-sampled on a pixel grid, so its discrete
    # energy is not perfectly proportional to contrast, and contrast varies along the kV axis.
    # The residual is ~1e-7 relative, four orders below anything the manuscript quotes.
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-5), "the floor is not linear in kV"
    assert floor[-1] / floor[0] == pytest.approx(kv[-1] / kv[0], rel=1e-5)


def test_no_placeholder_survives_into_a_submission_draft():
    """The submitted PDF must carry no placeholder token, and must name the repository.

    An unminted DOI is handled by saying so in prose — the archived release is inserted at
    submission — rather than by leaving a marker in the text that a reviewer would see.
    """
    text = SOURCE.read_text(encoding="utf-8")
    for placeholder in ("PENDING-ZENODO-DOI", "TODO", "TBD", "XXX"):
        assert placeholder not in text, f"{placeholder!r} is still in the manuscript"
    # The journal requires code to be reachable at review; promising it later is not enough.
    # (Saying the repository is available "rather than upon acceptance" is the point, so the
    # check is on the claim, not on the words.)
    for evasion in ("available upon acceptance", "available on request", "upon reasonable request"):
        assert evasion not in text.lower(), f"{evasion!r} is not an acceptable availability claim"
    # The journal's required back matter, in the sections it requires them in.
    for heading in ("## Disclosures", "## Code and Data Availability", "## Acknowledgments"):
        assert heading in text, f"{heading!r} is missing from the manuscript"
    availability = text.split("## Code and Data Availability", 1)[1].split("## Ack", 1)[0]
    assert "[[release:repository]]" in availability
    assert "[[release:archive_statement]]" in availability
    assert "MIT" in availability
    assert "available at review time" in availability
    disclosures = text.split("## Disclosures", 1)[1].split("## Code and Data", 1)[0]
    assert "conflicts of interest" in disclosures
    assert "Generative AI tools" in disclosures


def test_path_resolution_rules():
    """The marker language itself, so a broken resolver cannot silently pass everything."""
    payload = {
        "a": {"b": [10, 20, 30]},
        "rows": [
            {"x": 1.0, "name": "raw", "v": 7},
            {"x": 2.0, "name": "tv", "v": 9},
        ],
        "labels": {"TV(0.4 sd)": {"gain": -1.5}},
    }
    resolve = build_manuscript.resolve
    assert resolve(payload, "a.b[1]") == 20
    assert resolve(payload, "a.b[-1]") == 30
    assert resolve(payload, "rows[name=tv].v") == 9
    assert resolve(payload, "rows[x=2.0].name") == "tv"
    assert resolve(payload, 'labels["TV(0.4 sd)"].gain') == -1.5
    with pytest.raises(build_manuscript.ResolutionError):
        resolve(payload, "rows[name=nope].v")
    with pytest.raises(build_manuscript.ResolutionError):
        resolve(payload, "a.missing")
