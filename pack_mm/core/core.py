# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 16-11-2024
# ©alin m elena,
"""pack molecules inside various shapes."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.build import molecule as build_molecule
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.helpers.mlip_calculators import choose_calculator
from numpy import cos, exp, pi, random, sin, sqrt


def random_point_in_sphere(c: (float, float, float), r: float) -> (float, float, float):
    """
    Generate a random point inside a sphere of radius r, centered at c.

    Parameters
    ----------
        c (tuple): The center of the sphere as (x, y, z).
        r (float): The radius of the sphere.

    Returns
    -------
        tuple: A point (x, y, z) inside the sphere.
    """
    rad = r * random.rand() ** (1 / 3)

    theta = random.uniform(0, 2 * pi)
    phi = random.uniform(0, pi)

    x = c[0] + rad * sin(phi) * cos(theta)
    y = c[1] + rad * sin(phi) * sin(theta)
    z = c[2] + rad * cos(phi)

    return (x, y, z)


def random_point_in_ellipsoid(
    d: (float, float, float), a: float, b: float, c: float
) -> (float, float, float):
    """
    Generate a random point inside an ellipsoid with axes a, b, c, centered at d.

    Parameters
    ----------
        d (tuple): The center of the ellipsoid as (x, y, z).
        a (float): The semi-axis length of the ellipsoid along the x-axis.
        b (float): The semi-axis length of the ellipsoid along the y-axis.
        c (float): The semi-axis length of the ellipsoid along the z-axis.

    Returns
    -------
        tuple: A point (x, y, z) inside the ellipsoid.
    """
    theta = random.uniform(0, 2 * pi)
    phi = random.uniform(0, pi)
    rad = random.rand() ** (1 / 3)

    x = d[0] + a * rad * sin(phi) * cos(theta)
    y = d[1] + b * rad * sin(phi) * sin(theta)
    z = d[2] + c * rad * cos(phi)

    return (x, y, z)


def random_point_in_box(
    d: (float, float, float), a: float, b: float, c: float
) -> (float, float, float):
    """
    Generate a random point inside a box with sides a, b, c, centered at d.

    Parameters
    ----------
        d (tuple): The center of the box as (x, y, z).
        a (float): The length of the box along the x-axis.
        b (float): The length of the box along the y-axis.
        c (float): The length of the box along the z-axis.

    Returns
    -------
        tuple: A point (x, y, z) inside the box.
    """
    x = d[0] + random.uniform(-a * 0.5, a * 0.5)
    y = d[1] + random.uniform(-b * 0.5, b * 0.5)
    z = d[2] + random.uniform(-c * 0.5, c * 0.5)

    return (x, y, z)


def random_point_in_cylinder(
    c: (float, float, float), r: float, h: float, d: str
) -> (float, float, float):
    """
    Generate a random point inside a cylinder with radius r and height h, centered at c.

    Parameters
    ----------
        c (tuple): The center of the cylinder as (x, y, z).
        r (float): The radius of the cylinder's base.
        h (float): The height of the cylinder.
        direction (str): direction along which cylinger is oriented

    Returns
    -------
        tuple: A point (x, y, z) inside the cylinder.
    """
    theta = random.uniform(0, 2 * pi)
    rad = r * sqrt(random.rand())

    if d == "z":
        z = c[2] + random.uniform(-h * 0.5, h * 0.5)
        x = c[0] + rad * cos(theta)
        y = c[1] + rad * sin(theta)
    elif d == "y":
        y = c[1] + random.uniform(-h * 0.5, h * 0.5)
        x = c[0] + rad * cos(theta)
        z = c[2] + rad * sin(theta)
    elif d == "x":
        x = c[0] + random.uniform(-h * 0.5, h * 0.5)
        y = c[1] + rad * sin(theta)
        z = c[2] + rad * cos(theta)

    return (x, y, z)


def validate_value(label, x):
    """Validate input value, and raise an exception."""
    if x is not None and x < 0.0:
        err = f"Invalid {label}, needs to be positive"
        print(err)
        raise Exception(err)


def pack_molecules(
    system: str | None,
    molecule: str,
    nmols: int,
    arch: str,
    model: str,
    device: str,
    where: str,
    center: tuple[float, float, float] | None,
    radius: float | None,
    height: float | None,
    a: float | None,
    b: float | None,
    c: float | None,
    seed: int,
    temperature: float,
    ntries: int,
    geometry: bool,
    fmax: float,
    ca: float,
    cb: float,
    cc: float,
) -> None:
    """
    Pack molecules into a system based on the specified parameters.

    Parameters
    ----------
        system (str): Path to the system file or name of the system.
        molecule (str): Path to the molecule file or name of the molecule.
        nmols (int): Number of molecules to insert.
        arch (str): Architecture for the calculator.
        model (str): Path to the model file.
        device (str): Device to run calculations on (e.g., "cpu" or "cuda").
        where (str): Region to insert molecules ("anywhere",
                     "sphere", "cylinderZ", etc.).
        center (Optional[Tuple[float, float, float]]): Center of the insertion region.
        radius (Optional[float]): Radius for spherical or cylindrical insertion.
        height (Optional[float]): Height for cylindrical insertion.
        a, b, c (Optional[float]): Parameters for box or ellipsoid insertion.
        seed (int): Random seed for reproducibility.
        temperature (float): Temperature in Kelvin for acceptance probability.
        ntries (int): Maximum number of attempts to insert each molecule.
        geometry (bool): Whether to perform geometry optimization after insertion.
        ca, cb, cc (float): Cell dimensions if system is empty.
    """
    kbt = temperature * 8.6173303e-5  # Boltzmann constant in eV/K
    validate_value("temperature", temperature)
    validate_value("radius", radius)
    validate_value("height", height)
    validate_value("fmax", fmax)
    validate_value("seed", seed)
    validate_value("box a", a)
    validate_value("box b", b)
    validate_value("box c", c)
    validate_value("ntries", ntries)
    validate_value("cell box a", ca)
    validate_value("cell box b", cb)
    validate_value("nmols", nmols)
    validate_value("cell box c", cc)

    random.seed(seed)

    try:
        sys = read(system)
        sysname = Path(system).stem
    except Exception:
        sys = Atoms(cell=[ca, cb, cc], pbc=[True, True, True])
        sysname = "empty"

    cell = sys.cell.lengths()

    # Print initial information
    print(f"Inserting {nmols} {molecule} molecules in {sysname}.")
    print(f"Using {arch} model {model} on {device}.")
    print(f"Insert in {where}.")

    # Set center of insertion region
    if center is None:
        center = (cell[0] * 0.5, cell[1] * 0.5, cell[2] * 0.5)
    else:
        center = tuple(ci * cell[i] for i, ci in enumerate(center))

    # Set parameters based on insertion region
    if where == "anywhere":
        a, b, c = 1, 1, 1
    elif where == "sphere":
        if radius is None:
            radius = min(cell) * 0.5
    elif where in ["cylinderZ", "cylinderY", "cylinderX"]:
        if radius is None:
            if where == "cylinderZ":
                radius = min(cell[0], cell[1]) * 0.5
            elif where == "cylinderY":
                radius = min(cell[0], cell[2]) * 0.5
            elif where == "cylinderX":
                radius = min(cell[2], cell[1]) * 0.5
        if height is None:
            height = 0.5
    elif where == "box":
        a, b, c = a or 1, b or 1, c or 1
    elif where == "ellipsoid":
        a, b, c = a or 0.5, b or 0.5, c or 0.5

    calc = choose_calculator(
        arch=arch, model_path=model, device=device, default_dtype="float64"
    )
    sys.calc = calc

    e = sys.get_potential_energy() if len(sys) > 0 else 0.0

    csys = sys.copy()
    for i in range(nmols):
        accept = False
        for _itry in range(ntries):
            mol = load_molecule(molecule)
            tv = get_insertion_position(where, center, cell, a, b, c, radius, height)
            mol = rotate_molecule(mol)
            mol.translate(tv)

            tsys = csys.copy() + mol.copy()
            tsys.calc = calc
            en = tsys.get_potential_energy()
            de = en - e

            acc = exp(-de / kbt)
            u = random.random()
            print(f"Old energy={e}, new energy={en}, {de=}, {acc=}, random={u}")

            if u <= acc:
                accept = True
                break

        if accept:
            csys = tsys.copy()
            e = en
            print(f"Inserted particle {i + 1}")
            write(f"{sysname}+{i + 1}{Path(molecule).stem}.cif", csys)
        else:
            print(f"Failed to insert particle {i + 1} after {ntries} tries")
            optimize_geometry(
                f"{sysname}+{i + 1}{Path(molecule).stem}.cif", device, arch, model, fmax
            )

    # Perform final geometry optimization if requested
    if geometry:
        optimize_geometry(
            f"{sysname}+{nmols}{Path(molecule).stem}.cif", device, arch, model, fmax
        )


def load_molecule(molecule: str):
    """Load a molecule from a file or build it."""
    try:
        return build_molecule(molecule)
    except KeyError:
        return read(molecule)


def get_insertion_position(
    where: str,
    center: tuple[float, float, float],
    cell: list[float],
    a: float,
    b: float,
    c: float,
    radius: float | None,
    height: float | None,
) -> tuple[float, float, float]:
    """Get a random insertion position based on the region."""
    if where == "sphere":
        return random_point_in_sphere(center, radius)
    if where == "box":
        return random_point_in_box(center, cell[0] * a, cell[1] * b, cell[2] * c)
    if where == "ellipsoid":
        return random_point_in_ellipsoid(center, cell[0] * a, cell[1] * b, cell[2] * c)
    if where in ["cylinderZ", "cylinderY", "cylinderX"]:
        axis = where[-1].lower()
        return random_point_in_cylinder(center, radius, cell[2] * height, axis)
    # now is anywhere
    return random.random(3) * [a, b, c] * cell


def rotate_molecule(mol):
    """Rotate a molecule randomly."""
    ang = random.random(3)
    mol.euler_rotate(
        phi=ang[0] * 360, theta=ang[1] * 180, psi=ang[2] * 360, center=(0.0, 0.0, 0.0)
    )
    return mol


def optimize_geometry(
    struct_path: str, device: str, arch: str, model: str, fmax: float
) -> float:
    """Optimize the geometry of a structure."""
    geo = GeomOpt(
        struct_path=struct_path,
        device=device,
        fmax=fmax,
        calc_kwargs={"model_paths": model, "default_dtype": "float64"},
        filter_kwargs={"hydrostatic_strain": True},
    )
    geo.run()
    write(f"{struct_path}-opt.cif", geo.struct)
    return geo.struct.get_potential_energy()
