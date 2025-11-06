"""Test cli for packmm."""

# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena,
from __future__ import annotations

from typer.testing import CliRunner

from pack_mm.cli.packmm import app
from tests.utils import strip_ansi_codes

runner = CliRunner()


def test_packmm_default_values(caplog):
    """Check values."""
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "nothing to do" in caplog.text


def test_packmm_config(tmp_path, caplog):
    """Check config file."""
    with open(tmp_path / "sphere.yml", "w", encoding="utf-8") as f:
        conf = """nmols: 1
molecule: CH4
cell-a: 10.0
cell-b: 10.0
cell-c: 10.0
where: sphere
geometry: False
"""
        print(conf, file=f)
    result = runner.invoke(app, ["--config", tmp_path / "sphere.yml"])
    assert result.exit_code == 0
    assert "nmols=1" in caplog.text


def test_packmm_custom_molecule(caplog):
    """Check molecule."""
    result = runner.invoke(app, ["--molecule", "CO2"])
    assert result.exit_code == 0
    assert "molecule='CO2'" in caplog.text


def test_packmm_custom_nmols(caplog):
    """Check nmols."""
    result = runner.invoke(app, ["--nmols", "-2"])
    assert result.exit_code == 1
    assert "nmols=-2" in caplog.text


def test_packmm_custom_ntries(caplog):
    """Check ntries."""
    result = runner.invoke(app, ["--ntries", "1"])
    assert result.exit_code == 0
    assert "ntries=1" in caplog.text


def test_packmm_custom_seed(caplog):
    """Check seed."""
    result = runner.invoke(app, ["--seed", "1234"])
    assert result.exit_code == 0
    assert "seed=1234" in caplog.text


def test_packmm_custom_every(caplog):
    """Check seed."""
    result = runner.invoke(app, ["--every", "10"])
    assert result.exit_code == 0
    assert "every=10" in caplog.text


def test_packmm_custom_insertion_method(caplog):
    """Check insertion."""
    result = runner.invoke(app, ["--where", "sphere"])
    assert result.exit_code == 0
    assert "where=sphere" in caplog.text


def test_packmm_custom_insert_strategy(caplog):
    """Check insertion."""
    result = runner.invoke(app, ["--insert-strategy", "hmc"])
    assert result.exit_code == 0
    assert "insert_strategy=hmc" in caplog.text


def test_packmm_custom_relax_strategy(caplog):
    """Check relax."""
    result = runner.invoke(app, ["--relax-strategy", "md"])
    assert result.exit_code == 0
    assert "relax_strategy=md" in caplog.text


def test_packmm_custom_center(caplog):
    """Check centre."""
    result = runner.invoke(app, ["--centre", "0.5,0.5,0.5"])
    assert result.exit_code == 0
    assert "centre='0.5,0.5,0.5'" in caplog.text


def test_packmm_custom_radius(caplog):
    """Check radius."""
    result = runner.invoke(app, ["--radius", "10.0"])
    assert result.exit_code == 0
    assert "radius=10.0" in caplog.text


def test_packmm_custom_height(caplog):
    """Check height."""
    result = runner.invoke(app, ["--height", "5.0"])
    assert result.exit_code == 0
    assert "height=5.0" in caplog.text


def test_packmm_mlip(caplog):
    """Check mlip."""
    result = runner.invoke(
        app, ["--arch", "mace", "--model", "some", "--device", "cuda"]
    )
    assert result.exit_code == 0
    assert "arch='mace'" in caplog.text
    assert "model='some'" in caplog.text
    assert "device='cuda'" in caplog.text


def test_packmm_out_path(caplog):
    """Check out_path."""
    result = runner.invoke(app, ["--out-path", "out"])
    assert result.exit_code == 0
    assert "out_path='out'" in caplog.text


def test_packmm_custom_box_dimensions(caplog):
    """Check box."""
    result = runner.invoke(app, ["--a", "30.0", "--b", "30.0", "--c", "30.0"])
    assert result.exit_code == 0
    assert "a=30.0" in caplog.text
    assert "b=30.0" in caplog.text
    assert "c=30.0" in caplog.text


def test_packmm_empty_box_dimensions(caplog):
    """Check box empty."""
    result = runner.invoke(
        app, ["--cell-a", "30.0", "--cell-b", "30.0", "--cell-c", "30.0"]
    )
    assert result.exit_code == 0
    assert "cell_a=30.0" in caplog.text
    assert "cell_b=30.0" in caplog.text
    assert "cell_c=30.0" in caplog.text


def test_packmm_custom_temperature(caplog):
    """Check temperature."""
    result = runner.invoke(app, ["--temperature", "400.0"])
    assert result.exit_code == 0
    assert "temperature=400.0" in caplog.text


def test_packmm_md_temperature(caplog):
    """Check md temperature."""
    result = runner.invoke(app, ["--md-temperature", "300.0"])
    assert result.exit_code == 0
    assert "md_temperature=300.0" in caplog.text


def test_packmm_md_timestep(caplog):
    """Check md temperature."""
    result = runner.invoke(app, ["--md-timestep", "1.0"])
    assert result.exit_code == 0
    assert "md_timestep=1.0" in caplog.text


def test_packmm_md_steps(caplog):
    """Check md steps."""
    result = runner.invoke(app, ["--md-steps", "10"])
    assert result.exit_code == 0
    assert "md_steps=10" in caplog.text


def test_packmm_custom_fmax(caplog):
    """Check fmax."""
    result = runner.invoke(app, ["--fmax", "0.05"])
    assert result.exit_code == 0
    assert "fmax=0.05" in caplog.text


def test_packmm_custom_threshold(caplog):
    """Check threshold."""
    result = runner.invoke(app, ["--threshold", "0.9"])
    assert result.exit_code == 0
    assert "threshold=0.9" in caplog.text


def test_packmm_with_dispersion(caplog):
    """Check dispersion."""
    result = runner.invoke(app, ["--dispersion"])
    assert result.exit_code == 0
    assert "dispersion=True" in caplog.text


def test_packmm_no_geometry_optimization(caplog):
    """Check optimisation."""
    result = runner.invoke(app, ["--no-geometry"])
    assert result.exit_code == 0
    assert "geometry=False" in caplog.text


def test_packmm_invalid_insertion_method():
    """Check insertion."""
    result = runner.invoke(app, ["--where", "invalid_method"])
    assert result.exit_code != 0
    assert "Invalid value for '--where'" in strip_ansi_codes(result.output)


def test_packmm_invalid_insertion_strategy():
    """Check insertion strategt."""
    result = runner.invoke(app, ["--insert-strategy", "invalid_method"])
    assert result.exit_code != 0
    assert "Invalid value for '--insert-strategy'" in strip_ansi_codes(result.output)


def test_packmm_invalid_relax_strategy():
    """Check insertion strategt."""
    result = runner.invoke(app, ["--relax-strategy", "invalid_method"])
    assert result.exit_code != 0
    assert "Invalid value for '--relax-strategy'" in strip_ansi_codes(result.output)


def test_packmm_invalid_md_steps(caplog):
    """Check md steps."""
    result = runner.invoke(app, ["--nmols", "1", "--md-steps", "-10"])
    assert result.exit_code != 0
    assert "Invalid MD steps" in caplog.text


def test_packmm_invalid_md_temperature(caplog):
    """Check md steps."""
    result = runner.invoke(app, ["--nmols", "1", "--md-temperature", "-10.0"])
    assert result.exit_code != 0
    assert "Invalid MD temperature" in caplog.text


def test_packmm_invalid_md_timestep(caplog):
    """Check md timestep."""
    result = runner.invoke(app, ["--nmols", "1", "--md-timestep", "-10.0"])
    assert result.exit_code != 0
    assert "Invalid MD timestep" in caplog.text


def test_packmm_invalid_centre_format(caplog):
    """Check centre."""
    result = runner.invoke(app, ["--nmols", "1", "--centre", "0.5,0.5"])
    assert result.exit_code != 0
    assert "Invalid centre" in caplog.text


def test_packmm_invalid_radius(caplog):
    """Check box radius."""
    result = runner.invoke(app, ["--nmols", "1", "--radius", "-10.0"])
    assert result.exit_code != 0
    assert "Invalid radius" in caplog.text


def test_packmm_invalid_height(caplog):
    """Check box height."""
    result = runner.invoke(app, ["--nmols", "1", "--height", "-5.0"])
    assert result.exit_code != 0
    assert "Invalid height" in caplog.text


def test_packmm_invalid_box_dimensions_a(caplog):
    """Check box dimension."""
    result = runner.invoke(app, ["--nmols", "1", "--a", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box a" in caplog.text


def test_packmm_invalid_box_dimensions_b(caplog):
    """Check box dimension."""
    result = runner.invoke(app, ["--nmols", "1", "--b", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box b" in caplog.text


def test_packmm_invalid_box_dimensions_c(caplog):
    """Check box dimension."""
    result = runner.invoke(app, ["--nmols", "1", "--c", "-30.0"])
    assert result.exit_code != 0
    assert "Invalid box c" in caplog.text


def test_packmm_invalid_temperature(caplog):
    """Check temperature."""
    result = runner.invoke(app, ["--nmols", "1", "--temperature", "-400.0"])
    assert result.exit_code != 0
    assert "Invalid temperature" in caplog.text


def test_packmm_invalid_fmax(caplog):
    """Check fmax."""
    result = runner.invoke(app, ["--nmols", "1", "--fmax", "-0.05"])
    assert result.exit_code != 0
    assert "Invalid fmax" in caplog.text


def test_packmm_invalid_ntries(caplog):
    """Check ntries."""
    result = runner.invoke(app, ["--nmols", "1", "--ntries", "-1"])
    assert result.exit_code != 0
    assert "Invalid ntries" in caplog.text
