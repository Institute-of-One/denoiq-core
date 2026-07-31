# Release and submission checklist

The manuscript cites an archived release of this repository. The DOI for that release does not
exist until the release is made, and nothing in this repository can invent one: `paper/release.json`
holds `version_doi: null` until it is minted, and `python paper/build_pdf.py --submission` refuses
to build while it is null. That is deliberate — a submission PDF must not carry a placeholder.

## 1. Before the release

```bash
python -m pytest                                   # the whole suite, all green
python -m ruff format --check . && python -m ruff check .
python -m mypy denoiq_core
python paper/build_manuscript.py --check           # both documents resolve from results/
```

Confirm the metadata that goes into the archive:

| file | field | value |
|---|---|---|
| `pyproject.toml` | `version` | `0.1.0` |
| `denoiq_core/__init__.py` | `__version__` | `0.1.0` |
| `CITATION.cff` | `version` | `0.1.0` |
| `.zenodo.json` | `version` | `0.1.0` |
| `paper/release.json` | `version`, `tag` | `0.1.0`, `v0.1.0` |

All five must agree; `tests/test_release_metadata.py` asserts it.

## 2. Cut the release

```bash
git tag -a v0.1.0 -m "denoiq-core 0.1.0 — manuscript submission"
git push origin v0.1.0
```

Then create the GitHub release for the tag. With the Zenodo–GitHub integration switched on for
`Institute-of-One/denoiq-core`, publishing the release archives it and mints two DOIs:

* the **concept DOI**, which always resolves to the newest version;
* the **version DOI** for `v0.1.0`, which is the one the manuscript cites.

## 3. Insert the DOI

Put both into `paper/release.json`:

```json
{
  "concept_doi": "10.5281/zenodo.XXXXXXX",
  "version_doi": "10.5281/zenodo.YYYYYYY"
}
```

Add the concept DOI to `CITATION.cff` (`identifiers:`) and `.zenodo.json`
(`related_identifiers`, relation `isVersionOf`) as their comments describe. Do **not** set a
top-level `doi` in `.zenodo.json`: that stops Zenodo's own versioning.

## 4. Build the submission

```bash
python paper/build_manuscript.py --submission      # fails if version_doi is still null
python paper/make_figures.py
python paper/build_pdf.py --submission             # -> paper/build/manuscript.pdf
python paper/build_pdf.py --submission --document supplementary
```

The manuscript's Code and Data Availability section then reads
"archived at Zenodo as version 0.1.0, doi:10.5281/zenodo.YYYYYYY" instead of the draft phrasing.
Check that the built PDF contains no `PENDING`, `TODO` or `XXX` — `pytest` asserts this too.

## 5. Submit

`paper/build/manuscript.pdf` is set in the JMI submission style: Times-metric serif at 12 pt on a
wide leading, superscript citations, `Fig. N` references, SPIE section numbering, page numbers,
and the required Disclosures / Code and Data Availability / Acknowledgments back matter.
`paper/build/supplementary.pdf` is the supplement (Tables S1–S4, Fig. S1).
