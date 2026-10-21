import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_texture_fitting_runs(tmp_path, monkeypatch, capsys):
    gif = tmp_path / "progress.gif"
    argv = ["texture_fitting.py", "--device", "cpu", "--iterations", "4", "--views", "2"]
    argv += ["--resolution", "32", "--texture-size", "16", "--segments", "12"]
    argv += ["--frames", "2", "--gif", str(gif)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(EXAMPLES / "texture_fitting.py"), run_name="__main__")
    assert gif.exists()
    assert "step     3" in capsys.readouterr().out
