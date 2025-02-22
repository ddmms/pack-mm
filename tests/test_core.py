"""Test cli for core."""

# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena,
from __future__ import annotations

from ase import Atoms
from ase.build import molecule as build_molecule
from ase.io import write
import numpy as np
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
    validate_value,
)


# Set a fixed seed for reproducibility in tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed."""
    random.seed(2042)


def test_random_point_in_sphere():
    """Test point in sphere."""
    center = (0, 0, 0)
    radius = 10.0
    point = random_point_in_sphere(center, radius)
    assert len(point) == 3
    distance = np.linalg.norm(np.array(point) - np.array(center))
    assert distance <= radius


def test_random_point_in_ellipsoid():
    """Test point in ellipsoid."""
    center = (0, 0, 0)
    a, b, c = 1.0, 2.0, 3.0
    point = random_point_in_ellipsoid(center, a, b, c)
    assert len(point) == 3
    x, y, z = point
    assert (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) <= 1.0


def test_random_point_in_box():
    """Test point in box."""
    center = (0, 0, 0)
    a, b, c = 1.0, 2.0, 3.0
    point = random_point_in_box(center, a, b, c)
    assert len(point) == 3
    x, y, z = point
    assert center[0] - a * 0.5 <= x <= center[0] + a * 0.5
    assert center[1] - b * 0.5 <= y <= center[1] + b * 0.5
    assert center[2] - c * 0.5 <= z <= center[2] + c * 0.5


def test_random_point_in_cylinder():
    """Test point in cylinder."""
    center = (0, 0, 0)
    radius = 1.0
    height = 2.0
    direction = "z"
    point = random_point_in_cylinder(center, radius, height, direction)
    assert len(point) == 3
    x, y, z = point
    assert x**2 + y**2 <= radius**2
    assert center[2] - height * 0.5 <= z <= center[2] + height * 0.5


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
    molecule.write(molecule_file)
    loaded_molecule = load_molecule(str(molecule_file))
    assert isinstance(loaded_molecule, Atoms)
    assert len(loaded_molecule) == 3  # H2O has 3 atoms


def test_load_molecule_from_name():
    """Test point in load molecule."""
    molecule = load_molecule("H2O")
    assert isinstance(molecule, Atoms)
    assert len(molecule) == 3  # H2O has 3 atoms


def test_get_insertion_position_sphere():
    """Test point in sphere."""
    center = (0, 0, 0)
    radius = 10.0
    point = get_insertion_position("sphere", center, radius=radius)
    assert len(point) == 3
    distance = np.linalg.norm(np.array(point) - np.array(center))
    assert distance <= radius


def test_rotate_molecule():
    """Test rotate molecule."""
    molecule = build_molecule("H2O")
    rotated_molecule = rotate_molecule(molecule)
    assert isinstance(rotated_molecule, Atoms)
    assert len(rotated_molecule) == 3  # H2O has 3 atoms


def test_optimize_geometry(tmp_path):
    """Test go."""
    # Create a temporary structure file
    molecule = build_molecule("H2O")
    molecule.set_cell([10, 10, 10])
    molecule.set_pbc([True, True, True])
    structure_file = tmp_path / "water.cif"
    write(structure_file, molecule)
    optimized_energy = optimize_geometry(
        str(structure_file),
        device="cpu",
        arch="mace_mp",
        model="medium-omat-0",
        fmax=0.01,
    )
    assert optimized_energy == pytest.approx(-13.759273983276572, abs=1.0e-8)


# Test pack_molecules with a simple case
def test_pack_molecules(tmp_path):
    """Test pack molecule."""
    # Create a temporary system file
    system = Atoms(
        "Ca", positions=[(5.0, 5.0, 5.0)], cell=[10, 10, 10], pbc=[True, True, True]
    )
    system_file = tmp_path / "system.cif"
    write(system_file, system)

    # Test packing molecules
    e = pack_molecules(
        system=str(system_file),
        molecule="H2O",
        nmols=2,
        arch="mace_mp",
        model="medium-omat-0",
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
    assert e == pytest.approx(-28.251229837533085, abs=1.0e-6)
