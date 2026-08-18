"""Draw the real low-dose CT arm from the same file the table is built from.

Section 3.6 reports the held-out comparison as a table and had no figure, while four
figures produced during the ldct-io runs sat unused in ``paper/results/``. Those are
from earlier runs with a different lesion and a different trial count, so placing one
beside the table would have put a figure and a table with different numbers on the same
page. This reads ``results/real_liver.json`` -- the file the table's markers resolve
against -- so the two cannot disagree.

    python paper/make_real_liver_figure.py   # -> paper/figures/fig9_real_liver.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results" / "real_liver.json"
OUT = PAPER / "figures" / "fig9_real_liver.png"

# Drawn at roughly the width it prints at, so nothing shrinks into illegibility.
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "figure.dpi": 300})

SHORT = {
    "none": "unprocessed",
    "gaussian 0.75 mm": "gauss 0.75",
    "gaussian 1.00 mm": "gauss 1.00",
    "tv 1x noise": "TV",
    "nlm 0.8x noise": "NLM",
    "CNN small (21k)": "CNN 21 k",
    "CNN large (1850k)": "CNN 1.85 M",
}


def main() -> int:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    held = data["held_out"]
    rows = held["rows"]
    ceiling = held["ceiling"]
    rho = held["spearman_psnr_vs_d_prime"]["rho"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # --- left: fidelity against task, which is the paper's claim in one picture ----
    # Six of the seven methods land within 1.3 dB of each other, so per-point labels
    # overlap into mush. A legend cannot overlap.
    style = {
        "none": ("o", "#333333"),
        "tv 1x noise": ("^", "#2ca02c"),
        "nlm 0.8x noise": ("v", "#9467bd"),
        "gaussian 0.75 mm": ("P", "#8c564b"),
        "gaussian 1.00 mm": ("X", "#e377c2"),
        "CNN small (21k)": ("s", "#ff7f0e"),
        "CNN large (1850k)": ("D", "#d62728"),
    }
    for r in rows:
        marker, colour = style[r["label"]]
        axL.scatter(
            r["psnr"],
            r["d_prime"],
            s=70,
            zorder=3,
            marker=marker,
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            label=SHORT.get(r["label"], r["label"]),
        )
    axL.axhline(ceiling, color="#333", ls="--", lw=1.1)
    axL.text(
        axL.get_xlim()[0],
        ceiling,
        " closed-form ceiling",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#333",
    )
    axL.set_xlabel("PSNR against the full-dose image [dB]")
    axL.set_ylabel(r"task $d'$")
    axL.set_title(f"fidelity against task (Spearman $\\rho$ = {rho:+.2f})")
    axL.set_ylim(0, ceiling * 1.14)
    axL.margins(x=0.10)
    axL.grid(alpha=0.3)
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=7.5, loc="lower left", frameon=True, framealpha=0.95, ncol=2)

    # --- right: what 86x the capacity bought, and what it cost ---------------------
    cap = data["capacity"]
    small, large = cap["small"], cap["large"]
    x = [0, 1]
    axR.plot(x, [small["psnr"], large["psnr"]], "s-", color="#d62728", label="PSNR")
    axR.set_ylabel("PSNR [dB]", color="#d62728")
    axR.tick_params(axis="y", labelcolor="#d62728")
    axR.set_xticks(x)
    axR.set_xticklabels(
        [f"{small['parameters']:,}\nparameters", f"{large['parameters']:,}\nparameters"],
        fontsize=8.5,
    )
    twin = axR.twinx()
    twin.plot(x, [small["d_prime"], large["d_prime"]], "o-", color="#4C78A8", label=r"$d'$")
    twin.set_ylabel(r"task $d'$", color="#4C78A8")
    twin.tick_params(axis="y", labelcolor="#4C78A8")
    # Both axes span only their own two points, so state the sizes rather than let
    # the scaling imply that a 0.04 dB gain and a 0.35 loss in d' are comparable.
    axR.set_title(
        f"{cap['parameter_ratio']:.0f}x the capacity: "
        f"PSNR {large['psnr'] - small['psnr']:+.2f} dB, "
        f"$d'$ {large['d_prime'] - small['d_prime']:+.2f}",
        fontsize=9.5,
    )
    axR.set_xlim(-0.35, 1.35)
    axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT}")
    print(f"  {len(rows)} methods, ceiling d' = {ceiling:.2f}, Spearman rho = {rho:+.3f}")
    print(
        f"  capacity {small['parameters']:,} -> {large['parameters']:,} "
        f"({cap['parameter_ratio']:.0f}x): PSNR {small['psnr']:.2f} -> {large['psnr']:.2f} dB, "
        f"d' {small['d_prime']:.2f} -> {large['d_prime']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
