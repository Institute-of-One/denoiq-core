# What was actually sent

One directory per submission, holding the files the journal received, exactly as they were
uploaded. Nothing here is regenerated and nothing here is a build output, which is the whole
point: `paper/build/` is owned by `paper/build_all.py`, which deletes anything it did not write,
so a record kept there would vanish on the next build of the next version.

A directory is named `<journal>_<the journal's own reference>` and is written once, at the moment
of submission. It is never edited afterwards, not even to correct something that was wrong in
what was sent -- what was sent is the fact being recorded. Corrections belong in the manuscript
and in the next submission.

| Directory | Sent | Outcome |
|---|---|---|
| `physica_medica_EJMP-S-26-01310` | 2026-10-03, Physica Medica, full length article | Rejected before review 2026-10-21: in scope, "lack of sufficient novelty" |

The source that produced the Physica Medica set is not recoverable. It was revised in place for
the next venue before any of it was committed, so these built files are the only record of that
version. Commit at the moment of submission, not weeks later.
