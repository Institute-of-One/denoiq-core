import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "paper" / "redlamp_console.py"


def test_console_builds_three_scenes_without_torch(tmp_path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--denoiser",
            "tv",
            "--n-trials",
            "40",
            "--results",
            str(results),
            "--figures",
            str(figures),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    data = json.loads((results / "redlamp_console.json").read_text(encoding="utf-8"))
    assert data["denoiser"] == "tv"
    assert data["provenance"]["quick"] is True
    assert [s["name"] for s in data["scenes"]] == ["green", "amber", "red"]
    for scene in data["scenes"]:
        assert scene["dose_level"] in {"green", "amber", "red"}
        assert scene["ceiling_d_prime"] > 0

    assert (figures / "redlamp_console.html").exists()
    assert (figures / "console" / "red_raw.png").exists()
