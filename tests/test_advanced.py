"""Advanced tests for packmm."""

# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 23-02-2025
# ©alin m elena,
from __future__ import annotations

from ase.io import read
import pytest
from typer.testing import CliRunner

from pack_mm.cli.packmm import app

runner = CliRunner()

err = 1.0e-8


def test_packmm_hmc(tmp_path):
    """Check values."""
    result = runner.invoke(
        app,
        [
            "--nmols",
            "2",
            "--model",
            "small-0b2",
            "--insert-strategy",
            "hmc",
            "--seed",
            "2042",
            "--cell-a",
            "10",
            "--cell-b",
            "10",
            "--cell-c",
            "10",
            "--out-path",
            tmp_path,
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "1H2O.cif").exists()
    assert (tmp_path / "2H2O.cif").exists()
    f = read(tmp_path / "2H2O.cif")
    assert f[0].position == pytest.approx([7.83420049, 6.21661184, 5.21814887], abs=err)
    assert (tmp_path / "2H2O-opt.cif").exists()
    f = read(tmp_path / "2H2O-opt.cif")
    assert f[0].position == pytest.approx([7.83120709, 6.21742874, 5.22597213], abs=err)
