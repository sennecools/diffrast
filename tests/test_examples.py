import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Script, arguments for a run of a few seconds on the CPU, the flag that names the output file,
# and a phrase the script prints when it has finished.
CASES = {
    "texture_fitting": (
        "--iterations 4 --views 2 --resolution 32 --texture-size 16 --segments 24 "
        "--frames 2 --gif-width 160",
        "--gif",
        "texture  psnr",
    ),
    "shape_fitting": (
        "--iterations 20 --views 2 --resolution 32 --subdivisions 2 --segments 12 "
        "--frames 4 --gif-width 240",
        "--gif",
        "flipped triangles",
    ),
    "pose_fitting": (
        "--hypotheses 8 --iterations 8 --refine 4 --resolution 32 --texture-size 32 "
        "--frames 4 --gif-width 480",
        "--gif",
        "has the lowest loss",
    ),
    "depth_peeling": (
        "--width 96 --height 60 --supersample 2 --num-layers 4 --segments 48 "
        "--subdivisions 1 --panel-size 48 --check-size 32 --check-gradients",
        "--layers",
        "pixels covered",
    ),
    "env_lighting": (
        "--iterations 4 --views 2 --resolution 32 --env-size 8 --segments 48 --frames 2 "
        "--gif-width 240",
        "--gif",
        "cube map  psnr",
    ),
    "gbuffer": ("--panel 64 --supersample 1 --texture-size 32", "--out", "the knot covers"),
    "antialias_gradients": ("--resolution 64 --zoom 8", "--out", "vertices receive gradient"),
}


@pytest.mark.parametrize("name", CASES)
def test_example_runs(name, tmp_path, monkeypatch, capsys):
    arguments, output_flag, phrase = CASES[name]
    output = tmp_path / ("out.gif" if output_flag == "--gif" else "out.png")
    argv = [f"{name}.py", "--device", "cpu", *arguments.split(), output_flag, str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")
    assert output.stat().st_size > 0
    assert phrase in capsys.readouterr().out
