"""Bring the prose to -ize spelling, which is what the title already uses.

Why this is a reviewed list and not a regex
-------------------------------------------
The obvious implementation -- substitute ``ise`` for ``ize`` at the end of a word -- is wrong in
this manuscript in four separate ways, each of which was in the text when this was written:

* ``noise``, ``denoising``, ``precise``, ``premise``, ``comprises`` have no -ize form at all; the
  letters are not a suffix.
* ``armwise``, ``bitwise``, ``pixelwise`` and ``family-wise`` end in *-wise*, which a pattern
  looking for ``...ise`` happily matches.
* ``revising`` comes from *revise*, which likewise has no -ize form.
* ``characteristic`` contains ``characteris`` without being the verb.

So every word that changes is named below. Anything not named is left alone, and the script
checks afterwards that nothing outside the list moved.

Four regions are protected, because they are not the author's prose:

* ``[[...]]`` build markers, whose text is a path into a results file -- ``spec.cnn.optimiser``
  is a JSON key, and renaming it in the marker would make the marker fail to resolve;
* backtick code spans, which name real identifiers;
* quoted names used as keys, ``"optimiser": "Adam"``, for the same reason: the key is written in
  one file and read in another, and the word never reaches the page -- what the document shows is
  the value;
* the reference list, which carries other people's published titles. Rewriting the spelling of a
  title that was published as ``...characterisation...`` misquotes it.

Run it, read the report, run it again to confirm it is idempotent::

    python paper/convert_spelling.py --check    # report what would change, write nothing
    python paper/convert_spelling.py            # convert
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every word that changes, spelled out. Capitalised forms are listed separately rather than
#: handled by a case-insensitive match, so that the list reads as exactly what it does.
MAPPING = {
    "realisation": "realization",
    "realisations": "realizations",
    "Realisation": "Realization",
    "Realisations": "Realizations",
    "realises": "realizes",
    "channelised": "channelized",
    "Channelised": "Channelized",
    "normalisation": "normalization",
    "Normalisation": "Normalization",
    "normalised": "normalized",
    "Normalised": "Normalized",
    "normalises": "normalizes",
    "normalising": "normalizing",
    "quantisation": "quantization",
    "Quantisation": "Quantization",
    "optimised": "optimized",
    "optimising": "optimizing",
    "optimiser": "optimizer",
    "characterised": "characterized",
    "Characterised": "Characterized",
    "stylised": "stylized",
    "Stylised": "Stylized",
    "summarise": "summarize",
    "summarised": "summarized",
    "generalised": "generalized",
    "regularised": "regularized",
    "stabilise": "stabilize",
    "minimising": "minimizing",
}

WORD = re.compile(r"\b(" + "|".join(sorted(MAPPING, key=len, reverse=True)) + r")\b")

#: Files whose prose is the paper's own -- which is not only the manuscript. ``figures.py``
#: carries axis labels and legend entries that the reader sees inside the figures, and
#: ``build_pdf.py`` holds the Table 1 and Table 2 captions, which ``build_docx`` imports and puts
#: into the submitted document. Leaving either alone would print "channelised" in Figure 5 and in
#: Table 2 while the body said "channelized".
TARGETS = [
    ROOT / "paper" / "manuscript.md",
    ROOT / "paper" / "supplementary.md",
    ROOT / "paper" / "README.md",
    ROOT / "paper" / "cover_letter.txt",
    ROOT / "paper" / "build_pdf.py",
    ROOT / "paper" / "build_real_liver_results.py",
    ROOT / "denoiq_core" / "figures.py",
]

MARKER = re.compile(r"\[\[[^\]]*\]\]")
CODE_SPAN = re.compile(r"`[^`\n]*`")
KEY = re.compile(r"""(['"])[A-Za-z_][A-Za-z_0-9]*\1\s*:""")
PLACEHOLDER = re.compile("\x00(\\d+)\x00")


def convert(text: str) -> tuple[str, list[str]]:
    """Return the converted text and the words that changed, protecting the three regions."""
    protected: list[str] = []

    def hide(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    # The reference list is split off entirely rather than masked, so a published title keeps the
    # spelling it was published with.
    body, separator, references = text.partition("## References")
    body = MARKER.sub(hide, body)
    body = CODE_SPAN.sub(hide, body)
    body = KEY.sub(hide, body)

    changed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        changed.append(match.group(1))
        return MAPPING[match.group(1)]

    body = WORD.sub(replace, body)

    # The masks nest: a marker is usually written inside a code span, as `[[results:...]]`, so the
    # span that gets masked second contains the placeholder of the marker masked first.
    # ``re.sub`` does not rescan what it inserts, so one pass restores the span and leaves the
    # marker's placeholder sitting in the text as literal NUL bytes -- the marker silently
    # disappears from the manuscript. Restore until no placeholder is left.
    for _ in range(len(protected) + 1):
        restored = PLACEHOLDER.sub(lambda m: protected[int(m.group(1))], body)
        if restored == body:
            break
        body = restored
    if PLACEHOLDER.search(body):
        raise RuntimeError("placeholders left after restoring; masking is deeper than expected")
    return body + separator + references, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    total = 0
    for path in TARGETS:
        original = path.read_text(encoding="utf-8")
        converted, changed = convert(original)
        total += len(changed)
        print(f"{path.relative_to(ROOT)}: {len(changed)} occurrences")
        for word in sorted(set(changed)):
            print(f"    {changed.count(word):3d}  {word} -> {MAPPING[word]}")

        # Nothing outside the list may have moved. Comparing word by word catches a protected
        # region that was restored in the wrong place as well as an unintended substitution.
        before = re.findall(r"[A-Za-z]+", original)
        after = re.findall(r"[A-Za-z]+", converted)
        if len(before) != len(after):
            print(f"  ABORT: {path.name} changed its word count", file=sys.stderr)
            return 1
        for old, new in zip(before, after, strict=True):
            if old != new and MAPPING.get(old) != new:
                print(f"  ABORT: {path.name} changed {old!r} to {new!r}", file=sys.stderr)
                return 1

        # Idempotence: the reference converter once read only the spelling it did not write, so it
        # quietly did nothing on a second run. Check rather than assume.
        again, changed_again = convert(converted)
        if changed_again or again != converted:
            print(f"  ABORT: {path.name} is not stable under a second pass", file=sys.stderr)
            return 1

        if not args.check and changed:
            path.write_text(converted, encoding="utf-8")

    print(f"\n{total} occurrences {'would change' if args.check else 'changed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
