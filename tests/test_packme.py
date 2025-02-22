# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
import pytest
from typer.testing import CliRunner
from pack_me.cli.packme import app, InsertionMethod
from tests.utils import strip_ansi_codes

runner = CliRunner()

def test_packme_default_values():
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "nothing to do" in strip_ansi_codes(result.output)

def test_packme_custom_molecule():
    result = runner.invoke(app, ["--molecule", "CO2"])
    assert result.exit_code == 0
    assert "CO2" in strip_ansi_codes(result.output)

def test_packme_custom_nmols():
    result = runner.invoke(app, ["--nmols", "1"])
    assert result.exit_code == 0
    assert "nmols=1" in strip_ansi_codes(result.output)

def test_packme_custom_ntries():
    result = runner.invoke(app, ["--ntries", "1"])
    assert result.exit_code == 0
    assert "ntries=1" in strip_ansi_codes(result.output)

def test_packme_custom_seed():
    result = runner.invoke(app, ["--seed", "1234"])
    assert result.exit_code == 0
    assert "seed=1234" in strip_ansi_codes(result.output)

def test_packme_custom_insertion_method():
    result = runner.invoke(app, ["--where", "sphere"])
    assert result.exit_code == 0
    assert "where=sphere" in strip_ansi_codes(result.output)

def test_packme_custom_center():
    result = runner.invoke(app, ["--centre", "0.5,0.5,0.5"])
    assert result.exit_code == 0
    assert "centre='0.5,0.5,0.5'" in strip_ansi_codes(result.output)

def test_packme_custom_radius():
    result = runner.invoke(app, ["--radius", "10.0"])
    assert result.exit_code == 0
    assert "radius=10.0" in strip_ansi_codes(result.output)

def test_packme_custom_height():
    result = runner.invoke(app, ["--height", "5.0"])
    assert result.exit_code == 0
    assert "height=5.0" in strip_ansi_codes(result.output)

def test_packme_mlip():
    result = runner.invoke(app, ["--arch", "mace", "--model", "some", "--device", "cuda"])
    assert result.exit_code == 0
    assert "arch='mace'" in strip_ansi_codes(result.output)
    assert "model='some'" in strip_ansi_codes(result.output)
    assert "device='cuda'" in strip_ansi_codes(result.output)

def test_packme_custom_box_dimensions():
    result = runner.invoke(app, ["--a", "30.0", "--b", "30.0", "--c", "30.0"])
    assert result.exit_code == 0
    assert "a=30.0" in strip_ansi_codes(result.output)
    assert "b=30.0" in strip_ansi_codes(result.output)
    assert "c=30.0" in strip_ansi_codes(result.output)

def test_packme_empty_box_dimensions():
    result = runner.invoke(app, ["--cell-a", "30.0", "--cell-b", "30.0", "--cell-c", "30.0"])
    assert result.exit_code == 0
    assert "cell_a=30.0" in strip_ansi_codes(result.output)
    assert "cell_b=30.0" in strip_ansi_codes(result.output)
    assert "cell_c=30.0" in strip_ansi_codes(result.output)

def test_packme_custom_temperature():
    result = runner.invoke(app, ["--temperature", "400.0"])
    assert result.exit_code == 0
    assert "temperature=400.0" in strip_ansi_codes(result.output)

def test_packme_custom_fmax():
    result = runner.invoke(app, ["--fmax", "0.05"])
    assert result.exit_code == 0
    assert "fmax=0.05" in strip_ansi_codes(result.output)

def test_packme_no_geometry_optimization():
    result = runner.invoke(app, ["--no-geometry"])
    assert result.exit_code == 0
    assert "geometry=False" in strip_ansi_codes(result.output)

def test_packme_invalid_insertion_method():
    result = runner.invoke(app, ["--where", "invalid_method"])
    assert result.exit_code != 0
    assert "Invalid value for '--where'" in strip_ansi_codes(result.output)

def test_packme_invalid_centre_format():
    result = runner.invoke(app, ["--nmols", "1", "--centre", "0.5,0.5"])
    assert result.exit_code != 0
    assert "Invalid centre" in strip_ansi_codes(result.output)

def test_packme_invalid_centre_value():
    result = runner.invoke(app, ["--nmols", "1", "--centre", "-0.6,0.5,0.5"])
    assert result.exit_code != 0
    assert "Invalid centre" in strip_ansi_codes(result.output)

def test_packme_invalid_radius():
    result = runner.invoke(app, ["--nmols", "1","--radius", "-10.0"])
    assert result.exit_code != 0
    assert "Invalid radius" in strip_ansi_codes(result.output)

def test_packme_invalid_height():
    result = runner.invoke(app, ["--nmols", "1","--height", "-5.0"])
    assert result.exit_code != 0
    assert "Invalid height" in strip_ansi_codes(result.output)

def test_packme_invalid_box_dimensions_a():
    result = runner.invoke(app, ["--nmols","1","--a", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box a" in strip_ansi_codes(result.output)

def test_packme_invalid_box_dimensions_b():
    result = runner.invoke(app, ["--nmols","1", "--b", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box b" in strip_ansi_codes(result.output)

def test_packme_invalid_box_dimensions_c():
    result = runner.invoke(app, ["--nmols","1", "--c", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box c" in strip_ansi_codes(result.output)

def test_packme_invalid_temperature():
    result = runner.invoke(app, ["--nmols", "1", "--temperature", "-400.0"])
    assert result.exit_code != 0
    assert "Invalid temperature" in strip_ansi_codes(result.output)

def test_packme_invalid_fmax():
    result = runner.invoke(app, ["--nmols","1","--fmax", "-0.05"])
    assert result.exit_code != 0
    assert "Invalid fmax" in strip_ansi_codes(result.output)

def test_packme_invalid_ntries():
    result = runner.invoke(app, ["--nmols","1","--ntries", "-1"])
    assert result.exit_code != 0
    assert "Invalid ntries" in strip_ansi_codes(result.output)
