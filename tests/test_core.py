"""Tests for core."""

# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena,
from __future__ import annotations

from ase import Atoms
from ase.build import molecule as build_molecule
from ase.io import write
from numpy import random
import pytest

from pack_mm.core.core import (
    get_insertion_position,
    load_molecule,
    optimize_geometry,
    pack_molecules,
    random_point_in_box,
    random_point_in_cylinder,
    random_point_in_ellipsoid,
    random_point_in_sphere,
    rotate_molecule,
    run_md_nve,
    save_the_day,
    set_defaults,
    validate_value,
)

err = 1.0e-8


# Set a fixed seed for reproducibility in tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed."""
    random.seed(2042)


def test_random_point_in_sphere():
    """Test point in sphere."""
    center = (2, 2, 2)
    radius = 8.0
    x, y, z = random_point_in_sphere(center, radius)
    assert x == pytest.approx(-3.299696236298196, abs=err)
    assert y == pytest.approx(-3.046619861327052, abs=err)
    assert z == pytest.approx(1.1884891239565165, abs=err)


def test_random_point_in_ellipsoid():
    """Test point in ellipsoid."""
    center = (0, 0, 0)
    a, b, c = 2.0, 2.0, 4.0
    x, y, z = random_point_in_ellipsoid(center, a, b, c)
    assert x == pytest.approx(0.27914659851849705, abs=err)
    assert y == pytest.approx(-1.4815802529721946, abs=err)
    assert z == pytest.approx(-1.2059956538672925, abs=err)


def test_random_point_in_box():
    """Test point in box."""
    center = (0, 0, 0)
    a, b, c = 1.0, 2.0, 3.0
    x, y, z = random_point_in_box(center, a, b, c)
    assert x == pytest.approx(0.2796391455704136, abs=err)
    assert y == pytest.approx(0.24221552377157196, abs=err)
    assert z == pytest.approx(0.10546160764197499, abs=err)


def test_random_point_in_cylinder():
    """Test point in cylinder."""
    center = (0, 0, 0)
    radius = 2.0
    height = 5.0

    direction = "z"
    x, y, z = random_point_in_cylinder(center, radius, height, direction)
    assert x == pytest.approx(0.29184067570623795, abs=err)
    assert y == pytest.approx(-1.5489545079008842, abs=err)
    assert z == pytest.approx(0.1757693460699583, abs=err)

    direction = "y"
    x, y, z = random_point_in_cylinder(center, radius, height, direction)
    assert x == pytest.approx(-1.417921218944154, abs=err)
    assert y == pytest.approx(-0.8295636754899149, abs=err)
    assert z == pytest.approx(-1.2828806053300128, abs=err)

    direction = "x"
    x, y, z = random_point_in_cylinder(center, radius, height, direction)
    assert x == pytest.approx(0.5063181926884202, abs=err)
    assert y == pytest.approx(-0.5839696769917422, abs=err)
    assert z == pytest.approx(1.10139156645954, abs=err)


def test_validate_value_positive():
    """Test point in test value."""
    validate_value("test_value", 1.0)  # Should not raise an exception


def test_validate_value_negative():
    """Test point in test value."""
    with pytest.raises(Exception, match="Invalid test_value, needs to be positive"):
        validate_value("test_value", -1.0)


def test_load_molecule_from_file(tmp_path):
    """Test point in load molecule."""
    molecule = build_molecule("H2O")
    molecule_file = tmp_path / "water.xyz"
    write(molecule_file, molecule)
    loaded_molecule = load_molecule(str(molecule_file))
    assert len(loaded_molecule) == 3


def test_load_molecule_from_name():
    """Test point in load molecule."""
    molecule = load_molecule("H2O")
    assert len(molecule) == 3


def test_get_insertion_position_sphere():
    """Test point in sphere."""
    center = (5, 5, 5)
    radius = 10.0
    x, y, z = get_insertion_position("sphere", center, radius=radius)
    assert x == pytest.approx(-1.624620295372745, abs=err)
    assert y == pytest.approx(-1.3082748266588151, abs=err)
    assert z == pytest.approx(3.9856114049456455, abs=err)


def test_get_insertion_position_box():
    """Test point in box."""
    center = (5, 5, 5)
    a = 10.0
    x, y, z = get_insertion_position("box", center, a=a, b=a, c=a)
    assert x == pytest.approx(7.796391455704136, abs=err)
    assert y == pytest.approx(6.21107761885786, abs=err)
    assert z == pytest.approx(5.351538692139917, abs=err)


def test_get_insertion_position_ellipsoid():
    """Test point in ellipsoid."""
    center = (5, 5, 5)
    x, y, z = get_insertion_position("ellipsoid", center, a=2.0, b=2.0, c=4.0)
    assert x == pytest.approx(5.279146598518497, abs=err)
    assert y == pytest.approx(3.5184197470278056, abs=err)
    assert z == pytest.approx(3.7940043461327075, abs=err)


def test_get_insertion_position_cylinder():
    """Test point in cylinder."""
    center = (5, 5, 5)
    x, y, z = get_insertion_position("cylinderZ", center, radius=3.0, height=10.0)
    assert x == pytest.approx(5.437761013559357, abs=err)
    assert y == pytest.approx(2.6765682381486737, abs=err)
    assert z == pytest.approx(5.351538692139917, abs=err)


def test_set_defaults_centre():
    """Test centre."""
    cell = [10, 10, 10]
    centre, _, _, _, _, _ = set_defaults(cell, centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert centre[1] == pytest.approx(5.0, abs=err)
    assert centre[2] == pytest.approx(5.0, abs=err)


def test_set_defaults_box():
    """Test box."""
    cell = [10, 10, 10]
    centre, a, b, c, _, _ = set_defaults(cell, where="box", b=5.0, centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert a == pytest.approx(10.0, abs=err)
    assert b == pytest.approx(5.0, abs=err)
    assert c == pytest.approx(10.0, abs=err)


def test_set_defaults_anywhere():
    """Test box."""
    cell = [10, 10, 10]
    centre, a, b, c, _, _ = set_defaults(cell, where="anywhere", centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert a == pytest.approx(10.0, abs=err)
    assert b == pytest.approx(10.0, abs=err)
    assert c == pytest.approx(10.0, abs=err)


def test_set_defaults_ellipsoid():
    """Test ellipsoid."""
    cell = [10, 10, 10]
    centre, a, b, c, _, _ = set_defaults(
        cell, where="ellipsoid", b=3.0, a=3.0, centre=None
    )
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert a == pytest.approx(3.0, abs=err)
    assert b == pytest.approx(3.0, abs=err)
    assert c == pytest.approx(5.0, abs=err)


def test_set_defaults_cylinder():
    """Test ellipsoid."""
    cell = [10, 12, 14]
    centre, _, _, _, radius, height = set_defaults(cell, where="cylinderZ", centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert radius == pytest.approx(5.0, abs=err)
    assert height == pytest.approx(7.0, abs=err)

    centre, _, _, _, radius, height = set_defaults(cell, where="cylinderX", centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert radius == pytest.approx(6.0, abs=err)
    assert height == pytest.approx(5.0, abs=err)

    centre, _, _, _, radius, height = set_defaults(cell, where="cylinderY", centre=None)
    assert centre[0] == pytest.approx(5.0, abs=err)
    assert radius == pytest.approx(5.0, abs=err)
    assert height == pytest.approx(6.0, abs=err)


def test_rotate_molecule():
    """Test rotate molecule."""
    molecule = build_molecule("H2O")
    a1 = molecule.get_angle(2, 0, 1)
    a2 = molecule.get_angle(2, 1, 0)

    rotated_molecule = rotate_molecule(molecule)
    assert a1 == pytest.approx(rotated_molecule.get_angle(2, 0, 1))
    assert a2 == pytest.approx(rotated_molecule.get_angle(2, 1, 0))


def test_optimize_geometry(tmp_path):
    """Test go."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    optimized_energy, _ = optimize_geometry(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        fmax=0.01,
        out_path=tmp_path,
        opt_cell=True,
    )
    assert optimized_energy == pytest.approx(-14.17098106193308, abs=err)


def test_optimize_geometry_dispersion(tmp_path):
    """Test go with dispersion."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    optimized_energy, _ = optimize_geometry(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        fmax=0.01,
        out_path=tmp_path,
        opt_cell=True,
        dispersion=True,
    )
    assert optimized_energy == pytest.approx(-14.180876622530265, abs=err)


def test_pack_molecules(tmp_path):
    """Test pack molecule."""
    system = Atoms(
        "Ca", positions=[(5.0, 5.0, 5.0)], cell=[10, 10, 10], pbc=[True, True, True]
    )
    system_file = tmp_path / "system.cif"
    write(system_file, system)

    e, _ = pack_molecules(
        system=str(system_file),
        molecule="H2O",
        nmols=2,
        arch="mace_mp",
        model="small-0b2",
        device="cpu",
        where="sphere",
        center=(5.0, 5.0, 5.0),
        radius=5.0,
        seed=2042,
        temperature=300,
        ntries=10,
        geometry=False,
        fmax=0.1,
        out_path=tmp_path,
    )

    assert (tmp_path / "system+1H2O.cif").exists()
    assert (tmp_path / "system+2H2O.cif").exists()
    assert e == pytest.approx(-29.21589570470306, abs=err)


def test_pack_molecules_atoms(tmp_path):
    """Test pack molecule."""
    system = Atoms(
        "Ca", positions=[(5.0, 5.0, 5.0)], cell=[10, 10, 10], pbc=[True, True, True]
    )

    e, _ = pack_molecules(
        system=system,
        molecule="H2O",
        nmols=2,
        arch="mace_mp",
        model="small-0b2",
        device="cpu",
        where="sphere",
        center=(5.0, 5.0, 5.0),
        radius=5.0,
        seed=2042,
        temperature=300,
        ntries=10,
        geometry=False,
        fmax=0.1,
        out_path=tmp_path,
    )

    assert e == pytest.approx(-29.21589570470306, abs=err)


def test_pack_molecules_2(tmp_path, capsys):
    """Test pack molecule."""
    system = Atoms(
        "Ca", positions=[(2.5, 2.5, 2.5)], cell=[5, 5, 5], pbc=[True, True, True]
    )
    system_file = tmp_path / "system.cif"
    write(system_file, system)

    e, _ = pack_molecules(
        system=str(system_file),
        molecule="H2O",
        nmols=3,
        arch="mace_mp",
        model="small-0b2",
        device="cpu",
        where="sphere",
        center=(2.5, 2.5, 2.5),
        radius=2.5,
        seed=2042,
        temperature=300,
        ntries=2,
        geometry=False,
        fmax=0.1,
        out_path=tmp_path,
    )
    captured = capsys.readouterr()

    assert "Failed to insert particle 3 after 2 tries" in captured.out
    assert e == pytest.approx(-47.194755808249454, abs=err)


def test_pack_molecules_atoms_dispersion(tmp_path):
    """Test pack molecule with dispersion."""
    system = Atoms(
        "Ca", positions=[(2.5, 2.5, 2.5)], cell=[5, 5, 5], pbc=[True, True, True]
    )

    e, _ = pack_molecules(
        system=system,
        molecule="H2O",
        nmols=3,
        arch="mace_mp",
        model="small-0b2",
        dispersion=True,
        device="cpu",
        where="sphere",
        center=(2.5, 2.5, 2.5),
        radius=2.5,
        seed=2042,
        temperature=300,
        ntries=10,
        geometry=False,
        fmax=0.1,
        out_path=tmp_path,
    )
    # Without dispersion: -42.386622937612316
    assert e == pytest.approx(-42.93461715053182, abs=err)


def test_save_the_day(tmp_path):
    """Test save the day."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    s = save_the_day(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        fmax=0.01,
        out_path=tmp_path,
    )
    assert s[0].position == pytest.approx([5.0, 4.99619815, 5.30704738], abs=err)


def test_save_the_day_md(tmp_path):
    """Test save the day."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    s = save_the_day(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        relax_strategy="md",
        md_timestep=1.0,
        md_steps=10.0,
        md_temperature=100.0,
    )
    assert s[0].position == pytest.approx([4.99684244, 5.00440785, 5.2987255], abs=err)


def test_save_the_day_dispersion(tmp_path):
    """Test save the day with dispersion."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    s = save_the_day(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        dispersion=True,
        fmax=0.01,
        out_path=tmp_path,
    )
    assert s[0].position == pytest.approx([4.99998938, 4.99618389, 5.30704305], abs=err)


def test_save_the_day_invalid(tmp_path):
    """Test save the day."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    s = save_the_day(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        relax_strategy="invalid_stratregy",
        md_timestep=1.0,
        md_steps=10.0,
        md_temperature=100.0,
    )
    assert s is None


def test_run_md(tmp_path):
    """Test md."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    s = run_md_nve(
        molecule,
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        timestep=1.0,
        steps=10.0,
        temp=100.0,
    )
    assert s[0].position == pytest.approx([4.99684244, 5.00440785, 5.2987255], abs=err)


def test_run_md_dispersion(tmp_path):
    """Test md with dispersion."""
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    molecule.center()
    s = run_md_nve(
        molecule,
        device="cpu",
        arch="mace_mp",
        model="small-0b2",
        dispersion=True,
        timestep=1.0,
        steps=10.0,
        temp=100.0,
    )
    assert s[0].position == pytest.approx([4.99684244, 5.00440795, 5.29872694], abs=err)
