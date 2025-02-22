# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 16-11-2024
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html

from ase.io import read,write
from ase.build import molecule as build_molecule
from janus_core.calculations.geom_opt import GeomOpt
from numpy import random,exp,sin,cos,pi,array,sqrt
from janus_core.helpers.mlip_calculators import choose_calculator
from pathlib import Path
import argparse
from ase import Atoms
from typing import Optional, Tuple, List
import typer

packme = typer.Typer()

def random_point_in_sphere(c: (float,float,float), r: float) -> (float,float,float):
    """
    Generates a random point inside a sphere of radius r, centered at c.

    Parameters:
        c (tuple): The center of the sphere as (x, y, z).
        r (float): The radius of the sphere.

    Returns:
        tuple: A point (x, y, z) inside the sphere.
    """
    # Generate a random radius r' from 0 to r
    rad = r * random.rand() ** (1/3)

    # Generate random angles
    theta = random.uniform(0, 2 * pi)  # Azimuthal angle
    phi = random.uniform(0, pi)        # Polar angle

    # Convert spherical to Cartesian coordinates
    x = c[0] + rad * sin(phi) * cos(theta)
    y = c[1] + rad * sin(phi) * sin(theta)
    z = c[2] + rad * cos(phi)

    return (x, y, z)

def random_point_in_ellipsoid(d: (float,float,float), a: float, b: float, c: float) -> (float,float,float):
    """
    Generates a random point inside an ellipsoid with axes a, b, c, centered at d.

    Parameters:
        d (tuple): The center of the ellipsoid as (x, y, z).
        a (float): The semi-axis length of the ellipsoid along the x-axis.
        b (float): The semi-axis length of the ellipsoid along the y-axis.
        c (float): The semi-axis length of the ellipsoid along the z-axis.

    Returns:
        tuple: A point (x, y, z) inside the ellipsoid.
    """
    # Generate random spherical coordinates
    theta = random.uniform(0, 2 * pi)  # Azimuthal angle
    phi = random.uniform(0, pi)        # Polar angle
    rad = random.rand() ** (1/3)      # Radial distance scaled to fit the volume

    # Map spherical coordinates to ellipsoid coordinates
    x = d[0] + a * rad * sin(phi) * cos(theta)
    y = d[1] + b * rad * sin(phi) * sin(theta)
    z = d[2] + c * rad * cos(phi)

    return (x, y, z)

def random_point_in_box(d: (float,float,float), a: float, b: float, c: float) -> (float,float,float):
    """
    Generates a random point inside a box with sides a, b, c, centered at d.

    Parameters:
        d (tuple): The center of the box as (x, y, z).
        a (float): The length of the box along the x-axis.
        b (float): The length of the box along the y-axis.
        c (float): The length of the box along the z-axis.

    Returns:
        tuple: A point (x, y, z) inside the box.
    """
    # Generate random x, y, z coordinates within the box
    x = d[0] + random.uniform(-a * 0.5, a * 0.5)
    y = d[1] + random.uniform(-b * 0.5, b * 0.5)
    z = d[2] + random.uniform(-c * 0.5, c * 0.5)

    return (x, y, z)

def random_point_in_cylinder(c: (float,float,float), r: float, h: float, d: str) -> (float,float,float):
    """
    Generates a random point inside a cylinder with radius r and height h, centered at c.

    Parameters:
        c (tuple): The center of the cylinder as (x, y, z).
        r (float): The radius of the cylinder's base.
        h (float): The height of the cylinder.
        direction (str): direction along which cylinger is oriented

    Returns:
        tuple: A point (x, y, z) inside the cylinder.
    """

    # Generate random polar coordinates for x and y inside the circle
    theta = random.uniform(0, 2 * pi)  # Random angle in radians
    rad = r * sqrt(random.rand())  # Random radius within the circle

    if d == 'z':
      z = c[2] + random.uniform(-h*0.5, h*0.5)
      x = c[0] + rad * cos(theta)
      y = c[1] + rad * sin(theta)
    elif d == 'y':
      y = c[1] + random.uniform(-h*0.5, h*0.5)
      x = c[0] + rad * cos(theta)
      z = c[2] + rad * sin(theta)
    elif d == 'x':
      x = c[0] + random.uniform(-h*0.5, h*0.5)
      y = c[1] + rad * sin(theta)
      z = c[2] + rad * cos(theta)

    return (x, y, z)


@packme.command()
def pack_system(
    system: Optional[str] = typer.Option(None, help="The original box in which you want to add particles. If not provided, an empty box will be created."),
    molecule: str = typer.Option("H2O", help="Name of the molecule to be processed, ASE-recognizable or ASE-readable file."),
    nmols: int = typer.Option(10, help="Target number of molecules to insert."),
    ntries: int = typer.Option(50, help="Maximum number of attempts to insert each molecule."),
    seed: int = typer.Option(2025, help="Random seed for reproducibility."),
    where: Literal["anywhere", "sphere", "box", "cylinderZ", "cylinderY", "cylinderX", "ellipsoid"] = typer.Option(
        "anywhere",
        help="How to insert the molecule. Choices: 'anywhere', 'sphere', 'box', 'cylinderZ', 'cylinderY', 'cylinderX', 'ellipsoid'.",
        case_sensitive=False,
    ),
    center: Optional[str] = typer.Option(None, help="Center of the insertion zone in fractional coordinates, e.g., '0.12,0.4,0.5'."),
    radius: Optional[float] = typer.Option(None, help="Radius of the sphere or cylinder in Å, depending on the insertion volume."),
    height: Optional[float] = typer.Option(None, help="Height of the cylinder in fractional coordinates."),
    a: Optional[float] = typer.Option(None, help="Side of the box or semi-axis of the ellipsoid, fractional, depends on the insertion method."),
    b: Optional[float] = typer.Option(None, help="Side of the box or semi-axis of the ellipsoid, fractional, depends on the insertion method."),
    c: Optional[float] = typer.Option(None, help="Side of the box or semi-axis of the ellipsoid, fractional, depends on the insertion method."),
    device: str = typer.Option("cpu", help="Device to run calculations on (e.g., 'cpu' or 'cuda')."),
    model: str = typer.Option("medium-omat-0", help="ML model to use."),
    arch: str = typer.Option("mace_mp", help="MLIP architecture to use."),
    temperature: float = typer.Option(300.0, help="Temperature for the Monte Carlo acceptance rule."),
    ca: float = typer.Option(10.0, help="Side of the empty box along the x-axis."),
    cb: float = typer.Option(10.0, help="Side of the empty box along the y-axis."),
    cc: float = typer.Option(10.0, help="Side of the empty box along the z-axis."),
    geometry: bool = typer.Option(True, help="Perform geometry optimization at the end."),
):
    """
    Pack molecules into a system based on the specified parameters.
    """
    # Convert center string to tuple if provided
    if center:
        center = tuple(map(float, center.split(',')))

    # Call the main packing function
    pack_molecules(
        system=system,
        molecule=molecule,
        nmols=nmols,
        arch=arch,
        model=model,
        device=device,
        where=where,
        center=center,
        radius=radius,
        height=height,
        a=a,
        b=b,
        c=c,
        seed=seed,
        T=temperature,
        ntries=ntries,
        geometry=geometry,
        ca=ca,
        cb=cb,
        cc=cc,
    )

insertions = ['anywhere','sphere','box','cylinderZ','cylinderY','cylinderX','ellipsoid']


def pack_molecules(
    system: Optional[str],
    molecule: str,
    nmols: int,
    arch: str,
    model: str,
    device: str,
    where: str,
    center: Optional[Tuple[float, float, float]],
    radius: Optional[float],
    height: Optional[float],
    a: Optional[float],
    b: Optional[float],
    c: Optional[float],
    seed: int,
    T: float,
    ntries: int,
    geometry: bool,
    ca: float,
    cb: float,
    cc: float,
) -> None:
    """
    Pack molecules into a system based on the specified parameters.

    Parameters:
        system (str): Path to the system file or name of the system.
        molecule (str): Path to the molecule file or name of the molecule.
        nmols (int): Number of molecules to insert.
        arch (str): Architecture for the calculator.
        model (str): Path to the model file.
        device (str): Device to run calculations on (e.g., "cpu" or "cuda").
        where (str): Region to insert molecules ("anywhere", "sphere", "cylinderZ", etc.).
        center (Optional[Tuple[float, float, float]]): Center of the insertion region.
        radius (Optional[float]): Radius for spherical or cylindrical insertion.
        height (Optional[float]): Height for cylindrical insertion.
        a, b, c (Optional[float]): Parameters for box or ellipsoid insertion.
        seed (int): Random seed for reproducibility.
        T (float): Temperature in Kelvin for acceptance probability.
        ntries (int): Maximum number of attempts to insert each molecule.
        geometry (bool): Whether to perform geometry optimization after insertion.
        ca, cb, cc (float): Cell dimensions if system is empty.
    """

    kbT = T * 8.6173303e-5  # Boltzmann constant in eV/K

    random.seed(seed)

    try:
        sys = read(system)
        sysname = Path(system).stem
    except Exception as e:
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
        print(f"{a=} {b=} {c=}")
    elif where == "sphere":
        if radius is None:
            radius = min(cell) * 0.5
        print(f"{radius=}")
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
        print(f"{radius=} {height=}")
    elif where == "box":
        a, b, c = a or 1, b or 1, c or 1
        print(f"{a=} {b=} {c=}")
    elif where == "ellipsoid":
        a, b, c = a or 0.5, b or 0.5, c or 0.5
        print(f"{a=} {b=} {c=}")

    # Initialize calculator
    calc = choose_calculator(arch=arch, model_path=model, device=device, default_dtype="float64")
    sys.calc = calc

    # Get initial energy
    e = sys.get_potential_energy() if len(sys) > 0 else 0.0

    # Insert molecules
    csys = sys.copy()
    for i in range(nmols):
        accept = False
        for itry in range(ntries):
            mol = load_molecule(molecule)
            tv = get_insertion_position(where, center, cell, a, b, c, radius, height)
            mol = rotate_molecule(mol)
            mol.translate(tv)

            tsys = csys.copy() + mol.copy()
            tsys.calc = calc
            en = tsys.get_potential_energy()
            de = en - e

            acc = exp(-de / kbT)
            u = random.random()
            print(f"Old energy={e}, new energy={en}, change={de}, acceptance={acc}, random={u}")

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
            optimize_geometry(f"{sysname}+{i + 1}{Path(molecule).stem}.cif", device, model, e)

    # Perform final geometry optimization if requested
    if geometry:
        optimize_geometry(f"{sysname}+{nmols}{Path(molecule).stem}.cif", device, model, e)


def load_molecule(molecule: str):
    """Load a molecule from a file or build it."""
    try:
        return build_molecule(molecule)
    except:
        return read(molecule)


def get_insertion_position(
    where: str,
    center: Tuple[float, float, float],
    cell: List[float],
    a: float,
    b: float,
    c: float,
    radius: Optional[float],
    height: Optional[float],
) -> Tuple[float, float, float]:
    """Get a random insertion position based on the region."""
    if where == "anywhere":
        return random.random(3) * [a, b, c] * cell
    elif where == "sphere":
        return random_point_in_sphere(center, radius)
    elif where == "box":
        return random_point_in_box(center, cell[0] * a, cell[1] * b, cell[2] * c)
    elif where == "ellipsoid":
        return random_point_in_ellipsoid(center, cell[0] * a, cell[1] * b, cell[2] * c)
    elif where in ["cylinderZ", "cylinderY", "cylinderX"]:
        axis = where[-1].lower()
        return random_point_in_cylinder(center, radius, cell[2] * height, axis)


def rotate_molecule(mol):
    """Rotate a molecule randomly."""
    ang = random.random(3)
    mol.euler_rotate(phi=ang[0] * 360, theta=ang[1] * 180, psi=ang[2] * 360, center=(0.0, 0.0, 0.0))
    return mol


def optimize_geometry(struct_path: str, device: str, model: str, e: float):
    """Optimize the geometry of a structure."""
    geo = GeomOpt(
        struct_path=struct_path,
        device=device,
        calc_kwargs={'model_paths': model, 'default_dtype': 'float64'},
        filter_kwargs={"hydrostatic_strain": True},
    )
    geo.run()
    write(f"{struct_path}-opt.cif", geo.struct)
    return geo.struct.get_potential_energy()


if __name__ == "__main__":
    packme()

