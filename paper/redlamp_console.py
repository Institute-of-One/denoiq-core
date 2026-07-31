r"""Build the red-lamp console: a capture-ready reliability panel from live measurements.

    python paper/redlamp_console.py                      # CNN scenes (needs the [dl] extra)
    python paper/redlamp_console.py --denoiser tv        # torch-free fallback

For three acquisition settings at a fixed kV it renders, per scene, a raw and a processed
tile and the whole red-lamp verdict, and writes:

* ``results/redlamp_console.json`` — every number the panel shows, machine-readable, so the
  rule of this repository still holds: no figure carries a hand-typed number.
* ``paper/figures/console/`` — the image tiles.
* ``paper/figures/redlamp_console.html`` — a self-contained page (tiles inlined as data URLs)
  to open in a browser: the interactive artefact, and a good supplementary figure. The
  manuscript's Figure 8 is **not** a screenshot of it —
  :func:`denoiq_core.figures.figure8_console_panel` redraws the same record at journal text
  width, so the figure cannot fall out of step with the numbers.

The lamp is the *input gauge*: the analytic ideal-observer detectability of the **unprocessed
input** against a prespecified requirement (Rose, ``d' = 5``), i.e. the operational floor of
:mod:`denoiq_core.redlamp`. That floor is a task requirement rather than a zero-information
boundary — information remains below it — but no post-processing can bring the requirement back
into reach, because processing cannot exceed its input's ceiling. The processed tile's own
verdict (erasure, fidelity-task discordance) is shown as a badge on top of it. Every number
comes from :func:`denoiq_core.redlamp.assess`; nothing here measures anything the sweeps do
not.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import taskiq_core
from matplotlib.patches import Circle  # noqa: E402

import denoiq_core
from denoiq_core.denoisers import denoise
from denoiq_core.physics import DEFAULT_MODEL, DEFAULT_PHANTOM, acquisition_params, make_trials
from denoiq_core.redlamp import DEFAULT_CRITERIA, assess

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent

KV = 120.0
SCENES: tuple[tuple[str, float], ...] = (("green", 200.0), ("amber", 60.0), ("red", 25.0))
STATUS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b"}
LABEL = {"green": "GREEN", "amber": "AMBER", "red": "RED"}
ICON = {"green": "✓", "amber": "!", "red": "✗"}
#: fixed grayscale window (background 0, lesion contrast ~20) so the doses stay comparable
VMIN, VMAX = -55.0, 75.0


def render_tile(image: Any, path: Path, *, marker: bool = True) -> None:
    """Save one grayscale tile with a dashed ring at the known lesion location."""
    cen = (DEFAULT_PHANTOM.size - 1) / 2.0
    radius_px = DEFAULT_PHANTOM.radius_mm / DEFAULT_PHANTOM.spacing
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=140)
    ax.imshow(image, cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="nearest")
    if marker:
        ax.add_patch(
            Circle(
                (cen, cen),
                radius_px + 1.5,
                fill=False,
                edgecolor="#39d98a",
                lw=1.4,
                ls=(0, (3, 2)),
                alpha=0.9,
            )
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="#0b0e14")
    plt.close(fig)


def data_url(path: Path) -> str:
    """A base64 ``data:`` URL for a PNG, so the page is self-contained."""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return "data:image/png;base64," + encoded


def measure_scenes(
    method: str,
    params: dict[str, Any] | None,
    *,
    n_trials: int,
    seed: int,
    tiles_dir: Path,
) -> dict[str, Any]:
    """Run the gauge for every scene and render its tiles; return the record for the panel."""
    threshold = DEFAULT_CRITERIA.d_prime_threshold
    scenes: list[dict[str, Any]] = []
    for name, mas in SCENES:
        acq = acquisition_params(KV, mas, model=DEFAULT_MODEL)
        # `label` keeps a checkpoint path out of the reason strings, so the panel and the
        # record it writes are identical on any machine.
        dose = assess(KV, mas, "none", n_trials=n_trials, seed=seed, label="unprocessed input")
        proc = assess(KV, mas, method, params, n_trials=n_trials, seed=seed, label=method.upper())
        m = proc.metrics
        trials = make_trials(KV, mas, n_trials=n_trials, seed=seed)
        raw_img = trials.present[0]
        proc_img = denoise(trials.present, method, **(params or {}))[0]
        raw_path = tiles_dir / f"{name}_raw.png"
        proc_path = tiles_dir / f"{name}_proc.png"
        render_tile(raw_img, raw_path)
        render_tile(proc_img, proc_path)
        floor_mas = mas * (threshold / m["ceiling_d_prime"]) ** 2
        efficiency = (m["task_d_prime"] / m["ceiling_d_prime"]) ** 2
        scenes.append(
            {
                "name": name,
                "kv": KV,
                "mas": mas,
                "relative_dose": acq.relative_dose,
                "noise_sd": acq.noise_sd,
                "contrast": acq.contrast,
                "dose_level": dose.level,
                "dose_reason": dose.reasons[0] if dose.reasons else "",
                "proc_level": proc.level,
                "proc_reason": proc.reasons[0] if proc.reasons else "",
                "ceiling_d_prime": m["ceiling_d_prime"],
                "floor_mas": floor_mas,
                "raw_task_d_prime": m["raw_task_d_prime"],
                "proc_task_d_prime": m["task_d_prime"],
                "raw_ssim": dose.metrics["ssim"],
                "proc_ssim": m["ssim"],
                "proc_psnr": m["psnr"],
                "proc_efficiency": efficiency,
                "proc_contrast_recovery": m["contrast_recovery"],
                "proc_false_structure_rate": m["false_structure_rate"],
                "raw_tile": data_url(raw_path),
                "proc_tile": data_url(proc_path),
            }
        )
    return {
        "kv": KV,
        "rose_threshold": threshold,
        "n_trials": n_trials,
        "seed": seed,
        "denoiser": method,
        "model": DEFAULT_MODEL.to_dict(),
        "phantom": DEFAULT_PHANTOM.to_dict(),
        "scenes": scenes,
        "provenance": {
            "denoiq_core": denoiq_core.__version__,
            "taskiq_core": taskiq_core.__version__,
            "quick": n_trials < 400,
        },
    }


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#0d0d0d; color:#fff; padding:28px 24px 20px;
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif; -webkit-font-smoothing:antialiased; }
.wrap { max-width:1140px; margin:0 auto; }
header.top h1 { font-size:20px; margin:0 0 4px; letter-spacing:.2px; }
header.top p { margin:0; color:#c3c2b7; font-size:13px; line-height:1.5; }
header.top .muted { color:#898781; }
.key { display:flex; gap:18px; flex-wrap:wrap; margin:14px 0 20px; font-size:12px; color:#c3c2b7; }
.key b { color:#fff; font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
@media (max-width:820px){ .grid { grid-template-columns:1fr; } }
.card { background:#1a1a19; border:1px solid rgba(255,255,255,.10);
  border-top:3px solid var(--edge); border-radius:12px; padding:16px 16px 14px;
  display:flex; flex-direction:column; }
.c-head { display:flex; gap:14px; align-items:center; }
.tl { display:flex; flex-direction:column; gap:6px; padding:7px 6px; background:#0d0d0d;
  border-radius:9px; border:1px solid rgba(255,255,255,.08); }
.tl-dot { width:13px; height:13px; border-radius:50%; display:block; }
.c-head-txt { display:flex; flex-direction:column; gap:3px; }
.c-badge { display:inline-flex; align-items:center; gap:6px; align-self:flex-start; font-size:11px;
  font-weight:700; letter-spacing:.6px; padding:3px 9px; border-radius:999px; border:1px solid; }
.c-cond { font-size:17px; font-weight:650; margin-top:3px; }
.c-dose { font-size:11.5px; color:#898781; }
.dprime { display:flex; align-items:baseline; gap:8px; margin:14px 0 12px; padding:9px 12px;
  background:#0d0d0d; border-radius:9px; border:1px solid rgba(255,255,255,.07); }
.dp-l { font-size:11.5px; color:#898781; }
.dp-v { font-size:24px; font-weight:700; font-variant-numeric:tabular-nums; }
.dp-rose { font-size:11px; color:#898781; margin-left:auto; text-align:right; }
.tiles { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.tile { margin:0; position:relative; }
.tile img { width:100%; display:block; border-radius:8px; background:#0b0e14;
  border:1px solid rgba(255,255,255,.10); }
.t-verdict { position:absolute; top:7px; right:7px; font-size:10px; font-weight:700;
  padding:2px 7px; border-radius:999px; border:1px solid; letter-spacing:.4px; }
.tile figcaption { display:flex; flex-direction:column; gap:1px; margin-top:6px; }
.t-name { font-size:11px; font-weight:650; color:#fff; letter-spacing:.4px; }
.t-stat { font-size:10.5px; color:#898781; font-variant-numeric:tabular-nums; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:1px; margin:13px 0 6px;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.07);
  border-radius:9px; overflow:hidden; }
.st { background:#1a1a19; padding:8px 10px; display:flex; flex-direction:column; gap:2px; }
.st-l { font-size:10px; color:#898781; letter-spacing:.3px; text-transform:uppercase; }
.st-v { font-size:15px; font-weight:650; font-variant-numeric:tabular-nums; }
.st-sub { font-size:11px; color:#898781; font-weight:400; }
.reason { font-size:11.5px; line-height:1.45; color:#c3c2b7; margin:8px 0 0; padding-left:10px;
  border-left:2px solid rgba(255,255,255,.14); }
.reason.money { color:#fff; border-left-color:#d03b3b; }
.reason.sub { color:#898781; margin-top:5px; }
footer { margin-top:20px; padding-top:14px; border-top:1px solid rgba(255,255,255,.08);
  font-size:11px; color:#898781; line-height:1.6; }
footer b { color:#c3c2b7; font-weight:600; }
.thesis { color:#fff; font-size:13px; margin:0 0 6px; }
"""


def _clean(reason: str) -> str:
    return reason.split(": ", 1)[1] if ": " in reason else reason


def _traffic_light(active: str) -> str:
    dots = []
    for level in ("red", "amber", "green"):
        col = STATUS[level]
        if level == active:
            style = f"background:{col};box-shadow:0 0 14px 2px {col};opacity:1"
        else:
            style = f"background:{col};opacity:.16"
        dots.append(f'<span class="tl-dot" style="{style}"></span>')
    return '<div class="tl">' + "".join(dots) + "</div>"


def _stat(label: str, value: str, sub: str = "", accent: str = "") -> str:
    style = f' style="color:{accent}"' if accent else ""
    sub_html = f'<span class="st-sub">{html.escape(sub)}</span>' if sub else ""
    return (
        f'<div class="st"><span class="st-l">{html.escape(label)}</span>'
        f'<span class="st-v"{style}>{html.escape(value)}{sub_html}</span></div>'
    )


def _card(sc: dict[str, Any], rose: float, proc_name: str) -> str:
    lv = sc["dose_level"]
    col = STATUS[lv]
    pcol = STATUS[sc["proc_level"]]
    money = "money" if lv == "red" else ""
    head = (
        f'<header class="c-head">{_traffic_light(lv)}<div class="c-head-txt">'
        f'<div class="c-badge" style="background:{col}1a;color:{col};border-color:{col}55">'
        f"<span>{ICON[lv]}</span> DOSE&nbsp;·&nbsp;{LABEL[lv]}</div>"
        f'<div class="c-cond">{sc["kv"]:.0f}&thinsp;kV&nbsp;·&nbsp;{sc["mas"]:.0f}&thinsp;mAs</div>'
        f'<div class="c-dose">relative dose ×{sc["relative_dose"]:.2f}'
        f"&nbsp;·&nbsp; noise σ {sc['noise_sd']:.0f}</div></div></header>"
    )
    dprime = (
        f'<div class="dprime"><span class="dp-l">input ideal d′</span>'
        f'<span class="dp-v" style="color:{col}">{sc["ceiling_d_prime"]:.2f}</span>'
        f'<span class="dp-rose">vs Rose {rose:.0f} · floor ≈ {sc["floor_mas"]:.0f} mAs</span></div>'
    )
    verdict = (
        f'<span class="t-verdict" style="background:{pcol}1a;color:{pcol};border-color:{pcol}55">'
        f"{ICON[sc['proc_level']]} {proc_name} {LABEL[sc['proc_level']]}</span>"
    )
    tiles = (
        '<div class="tiles">'
        f'<figure class="tile"><img src="{sc["raw_tile"]}" alt="raw"/>'
        f'<figcaption><span class="t-name">UNPROCESSED</span><span class="t-stat">'
        f"SSIM {sc['raw_ssim']:.2f} · task d′ {sc['raw_task_d_prime']:.2f}"
        "</span></figcaption></figure>"
        f'<figure class="tile"><img src="{sc["proc_tile"]}" alt="processed"/>{verdict}'
        f'<figcaption><span class="t-name">{proc_name} denoised</span><span class="t-stat">'
        f"SSIM {sc['proc_ssim']:.2f} · task d′ {sc['proc_task_d_prime']:.2f}"
        "</span></figcaption></figure></div>"
    )
    stats = (
        '<div class="stats">'
        + _stat(f"{proc_name} SSIM / PSNR", f"{sc['proc_ssim']:.2f} / {sc['proc_psnr']:.0f} dB")
        + _stat(
            f"{proc_name} task d′",
            f"{sc['proc_task_d_prime']:.2f}",
            sub=f"  (raw {sc['raw_task_d_prime']:.2f})",
        )
        + _stat("information kept", f"{sc['proc_efficiency'] * 100:.0f}%", accent=pcol)
        + _stat("contrast recovered", f"{sc['proc_contrast_recovery'] * 100:.0f}%")
        + "</div>"
    )
    reasons = (
        f'<p class="reason {money}">{html.escape(_clean(sc["dose_reason"]))}</p>'
        f'<p class="reason sub">{proc_name}: {html.escape(_clean(sc["proc_reason"]))}</p>'
    )
    return (
        f'<section class="card" style="--edge:{col}">'
        f"{head}{dprime}{tiles}{stats}{reasons}</section>"
    )


def build_html(data: dict[str, Any]) -> str:
    """Assemble the self-contained console page from a measured record."""
    rose = data["rose_threshold"]
    proc_name = "CNN" if data["denoiser"].lower().startswith("cnn") else data["denoiser"].upper()
    last = data["scenes"][-1]
    cards = "".join(_card(sc, rose, proc_name) for sc in data["scenes"])
    intro = (
        f'<header class="top"><h1>Red-Lamp Console — Information-Limited CT Denoising</h1>'
        f"<p>Synthetic demonstration on a signal-known-exactly detection task "
        f"({data['kv']:.0f}&thinsp;kV, {data['phantom']['radius_mm']:.0f}&thinsp;mm lesion). "
        f'The lamp is the <b style="color:#fff">input gauge</b>: the analytic ideal-observer '
        f'detectability of the <b style="color:#fff">unprocessed input</b> against a '
        f"prespecified requirement (Rose, d′&nbsp;=&nbsp;{rose:.0f}) — the operational floor. "
        f"It is a task requirement, not a zero-information boundary: information remains below "
        f"it, but no post-processing can bring the requirement back into reach. "
        f'<span class="muted">A method and its demonstration, not a scanner calibration — the '
        f"coefficients are a documented relative model.</span></p></header>"
    )
    key = (
        '<div class="key">'
        "<span>\U0001f7e2 <b>GREEN</b> input meets the requirement</span>"
        "<span>\U0001f7e1 <b>AMBER</b> marginal (within 1.2× of the floor)</span>"
        "<span>\U0001f534 <b>RED</b> input below the requirement</span>"
        f"<span>· <b>{proc_name} badge</b> = what the processing did "
        "(erasure / fidelity-task discordance)</span></div>"
    )
    footer = (
        f'<footer><p class="thesis">The cleanest image can carry the least task information: at '
        f"{last['mas']:.0f}&thinsp;mAs the {proc_name} reaches SSIM&nbsp;{last['proc_ssim']:.2f} "
        f"yet a held-out prewhitening task d′ of only {last['proc_task_d_prime']:.2f} — against "
        f"{last['raw_task_d_prime']:.2f} on the unprocessed input it was cleaned from.</p>"
        f"<b>Provenance:</b> denoiq-core {data.get('version', '0.1.0')} · "
        f"{data['n_trials']} trials per class per condition · seed {data['seed']} · every number "
        "computed live by denoiq_core.redlamp.assess. Lamp: analytic ideal linear (prewhitening) "
        "observer on the unprocessed input, from its analytic NPS. Tiles: held-out prewhitening "
        "linear observer. Institute of One, LISIT Co., Ltd., Tokyo, Japan.</footer>"
    )
    return (
        '<!doctype html>\n<html lang="en" data-theme="dark">\n<head>\n'
        '<meta charset="utf-8"/>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        "<title>Red-Lamp Console — Information-Limited CT Denoising</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div class="wrap">{intro}{key}<div class="grid">{cards}</div>{footer}</div>\n'
        "</body>\n</html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Regenerate the console JSON, tiles and HTML."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=REPO_DIR / "results")
    parser.add_argument("--figures", type=Path, default=PAPER_DIR / "figures")
    parser.add_argument(
        "--denoiser",
        default="cnn",
        help="processed denoiser: 'cnn' (needs [dl]) or a classical method (tv/gaussian/nlm)",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_DIR / "checkpoints" / "cnn.pt")
    parser.add_argument("--n-trials", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.denoiser == "cnn":
        import denoiq_core.cnn as cnn  # noqa: F401 - registers the 'cnn' denoiser

        if not cnn.torch_available():
            parser.error("--denoiser cnn needs the [dl] extra; use --denoiser tv without it")
        if not args.checkpoint.exists():
            parser.error(
                f"no checkpoint at {args.checkpoint}; train one with 'python -m denoiq_core.train'"
            )
        params: dict[str, Any] | None = {"checkpoint": str(args.checkpoint)}
    else:
        params = None

    tiles_dir = args.figures / "console"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    data = measure_scenes(
        args.denoiser, params, n_trials=args.n_trials, seed=args.seed, tiles_dir=tiles_dir
    )

    if args.denoiser == "cnn":
        from denoiq_core.cnn import checkpoint_sha256

        data["checkpoint_sha256"] = checkpoint_sha256(args.checkpoint)

    json_path = args.results / "redlamp_console.json"
    record = {k: v for k, v in data.items()}
    record["scenes"] = [
        {k: v for k, v in sc.items() if not k.endswith("_tile")} for sc in data["scenes"]
    ]
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    html_path = args.figures / "redlamp_console.html"
    html_path.write_text(build_html(data), encoding="utf-8")

    print(f"tiles:   {tiles_dir}")
    print(f"numbers: {json_path}")
    print(f"console: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
