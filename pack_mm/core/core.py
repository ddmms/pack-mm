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
from ase.units import kB
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.md import NVE
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


def validate_value(label: str, x: float | int) -> None:
    """Validate input value, and raise an exception."""
    if x is not None and x < 0.0:
        err = f"Invalid {label}, needs to be positive"
        print(err)
        raise Exception(err)


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    random.seed(seed)


def set_defaults(
    cell: (float, float, float),
    centre: (float, float, float) | None = None,
    where: str | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    radius: float | None = None,
    height: float | None = None,
) -> tuple(
    (float, float, float),
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
):
    """Set defaults for insertion areas."""
    if centre is None:
        centre = (cell[0] * 0.5, cell[1] * 0.5, cell[2] * 0.5)

    if where == "anywhere":
        a, b, c = cell[0], cell[1], cell[2]
    elif where == "sphere":
        radius = radius or min(cell) * 0.5
    elif where == "cylinderZ":
        radius = radius or min(cell[0], cell[1]) * 0.5
        height = height or 0.5 * cell[2]
    elif where == "cylinderY":
        radius = radius or min(cell[0], cell[2]) * 0.5
        height = height or 0.5 * cell[1]
    elif where == "cylinderX":
        radius = radius or min(cell[2], cell[1]) * 0.5
        height = height or 0.5 * cell[0]
    elif where == "box":
        a, b, c = a or cell[0], b or cell[1], c or cell[2]
    elif where == "ellipsoid":
        a, b, c = a or cell[0] * 0.5, b or cell[1] * 0.5, c or cell[2] * 0.5
    return (centre, a, b, c, radius, height)


def pack_molecules(
    system: str | Atoms = None,
    molecule: str = "H2O",
    nmols: int = -1,
    arch: str = "mace_mp",
    model: str = "medium-omat-0",
    dispersion: bool = False,
    device: str = "cpu",
    where: str = "anywhere",
    center: tuple[float, float, float] = None,
    radius: float = None,
    height: float = None,
    a: float = None,
    b: float = None,
    c: float = None,
    seed: int = 2025,
    temperature: float = 300.0,
    ntries: int = 50,
    geometry: bool = False,
    fmax: float = 0.1,
    threshold: float = 0.92,
    cell_a: float = None,
    cell_b: float = None,
    cell_c: float = None,
    out_path: str = ".",
    every: int = -1,
    relax_strategy: str = "geometry_optimisation",
    insert_strategy: str = "mc",
    md_steps: int = 10,
    md_timestep: float = 1.0,
    md_temperature: float = 100.0,
) -> tuple(float, Atoms):
    """
    Pack molecules into a system based on the specified parameters.

    Parameters
    ----------
        system (str|Atoms): Path to the system file or name of the system.
        molecule (str): Path to the molecule file or name of the molecule.
        nmols (int): Number of molecules to insert.
        arch (str): Architecture for the calculator.
        model (str): Path to the model file.
        dispersion (bool): If a dispersion correction is added to MLIP calculator.
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
        cell_a, cell_b, cell_c (float): Cell dimensions if system is empty.
        out_path (str): path to save various outputs
        every (int): After how many instertions to do a relaxation,
                     default -1 means none..
        md_temperature (float): Temperature in Kelvin for MD.
        md_steps (int): Number of steps for MD.
        md_timestep (float): Timestep in fs for MD.
        fmax (float): Max force for geometry optimisation.
        threshold (float): Percentage of the single molecule energy above which
                           the move is to be considered for acceptance.
        insert_strategy (str): Insert strategy, "random" or "md"
        relax_strategy (str): Relax strategy, "geometry_optimisation" or "md"

    Returns
    -------
        tuple: A tuple energy and Atoms object containing
               original system and added molecules..

    """
    kbt = temperature * kB
    validate_value("temperature", temperature)
    validate_value("radius", radius)
    validate_value("height", height)
    validate_value("fmax", fmax)
    validate_value("seed", seed)
    validate_value("box a", a)
    validate_value("box b", b)
    validate_value("box c", c)
    validate_value("ntries", ntries)
    validate_value("cell box cell a", cell_a)
    validate_value("cell box cell b", cell_b)
    validate_value("cell box cell c", cell_c)
    validate_value("nmols", nmols)
    validate_value("MD steps", md_steps)
    validate_value("MD timestep", md_timestep)
    validate_value("MD temperature", md_temperature)

    set_random_seed(seed)

    if system is None:
        sys = Atoms(cell=[cell_a, cell_b, cell_c], pbc=[True, True, True])
        sysname = ""
    elif isinstance(system, Atoms):
        sys = system.copy()
        sysname = sys.get_chemical_formula() + "+"
    else:
        sys = read(system)
        sysname = Path(system).stem + "+"

    # Print summary
    print(f"Inserting {nmols} {molecule} molecules in {sysname}.")
    print(f"Using {arch} model {model} on {device}.")
    print(f"Insert in {where}.")

    cell = sys.cell.lengths()

    center, a, b, c, radius, height = set_defaults(
        cell, center, where, a, b, c, radius, height
    )

    calc = choose_calculator(
        arch=arch,
        model_path=model,
        device=device,
        dispersion=dispersion,
    )
    sys.calc = calc

    e = sys.get_potential_energy() if len(sys) > 0 else 0.0

    mol = load_molecule(molecule)
    mol.calc = calc
    emol = mol.get_potential_energy()

    csys = sys.copy()
    i = 0
    while i < nmols:
        accept = False
        for _itry in range(ntries):
            mol = load_molecule(molecule)
            tv = get_insertion_position(where, center, a, b, c, radius, height)
            mol = rotate_molecule(mol)
            mol.translate(tv)

            tsys = csys.copy() + mol.copy()
            if insert_strategy == "hmc":
                tsys = run_md_nve(
                    tsys,
                    md_temperature,
                    md_steps,
                    md_timestep,
                    arch,
                    model,
                    dispersion,
                    device,
                )

            if every > 0 and _itry / every == 0:
                tsys = save_the_day(
                    struct=tsys,
                    device=device,
                    arch=arch,
                    model=model,
                    dispersion=dispersion,
                    fmax=fmax,
                    out_path=out_path,
                    md_temperature=md_temperature,
                    md_steps=md_steps,
                    md_timestep=md_timestep,
                    relax_strategy=relax_strategy,
                )

            tsys.calc = calc
            en = tsys.get_potential_energy()
            de = en - e

            acc = exp(-de / kbt)
            u = random.random()
            print(f"Old energy={e}, new energy={en}, {de=}, {acc=}, random={u}")

            if abs(de / emol) > threshold and u <= acc:
                accept = True
                break
        if accept:
            csys = tsys.copy()
            e = en
            i += 1
            print(f"Inserted particle {i}")
            write(Path(out_path) / f"{sysname}{i}{Path(molecule).stem}.cif", csys)
        else:
            # Things are bad, maybe geomatry optimisation saves us
            # once you hit here is bad, this can keep looping
            print(f"Failed to insert particle {i + 1} after {ntries} tries")
            csys = save_the_day(
                csys,
                device,
                arch,
                model,
                dispersion,
                fmax,
                out_path,
                md_temperature,
                md_steps,
                md_timestep,
                relax_strategy,
            )

    energy_final = e

    # Perform final geometry optimization if requested
    if geometry:
        energy_final, csys = optimize_geometry(
            struct=Path(out_path) / f"{sysname}{nmols}{Path(molecule).stem}.cif",
            device=device,
            arch=arch,
            model=model,
            fmax=fmax,
            out_path=out_path,
            opt_cell=True,
            dispersion=dispersion,
        )
    return (energy_final, csys)


def load_molecule(molecule: str):
    """Load a molecule from a file or build it."""
    try:
        return build_molecule(molecule)
    except KeyError:
        return read(molecule)


def get_insertion_position(
    where: str,
    center: tuple[float, float, float],
    a: float = None,
    b: float = None,
    c: float = None,
    radius: float = None,
    height: float = None,
) -> tuple[float, float, float]:
    """Get a random insertion position based on the region."""
    if where == "sphere":
        return random_point_in_sphere(center, radius)
    if where == "box":
        return random_point_in_box(center, a, b, c)
    if where == "ellipsoid":
        return random_point_in_ellipsoid(center, a, b, c)
    if where in ["cylinderZ", "cylinderY", "cylinderX"]:
        axis = where[-1].lower()
        return random_point_in_cylinder(center, radius, height, axis)
    return random.random(3) * [a, b, c]


def rotate_molecule(mol):
    """Rotate a molecule randomly."""
    ang = random.random(3)
    mol.euler_rotate(
        phi=ang[0] * 360, theta=ang[1] * 180, psi=ang[2] * 360, center=(0.0, 0.0, 0.0)
    )
    return mol


def save_the_day(
    struct: str | Atoms,
    device: str = "",
    arch: str = "",
    model: str = "",
    dispersion: bool = False,
    fmax: float = 0.01,
    out_path: str = ".",
    md_temperature: float = 100.0,
    md_steps: int = 10,
    md_timestep: float = 1.0,
    relax_strategy: str = "geometry_optimisation",
) -> Atoms:
    """Geometry optimisation or MD to get a better structure."""
    if relax_strategy == "geometry_optimisation":
        _, a = optimize_geometry(
            struct,
            device,
            arch,
            model,
            fmax,
            out_path,
            dispersion=dispersion,
        )
        return a
    if relax_strategy == "md":
        return run_md_nve(
            struct,
            md_temperature,
            md_steps,
            md_timestep,
            arch,
            model,
            dispersion,
            device,
        )
    return None


def run_md_nve(
    struct: str | Atoms,
    temp: float = 100.0,
    steps: int = 10,
    timestep: float = 1.0,
    arch: str = "",
    model: str = "",
    dispersion: bool = False,
    device: str = "",
) -> Atoms:
    """Run nve simulation."""
    md = NVE(
        struct=struct,
        temp=temp,
        device=device,
        arch=arch,
        calc_kwargs={"model_paths": model, "dispersion": dispersion},
        stats_every=1,
        steps=steps,
        timestep=timestep,
    )
    md.run()
    return md.struct


def optimize_geometry(
    struct: str | Atoms,
    device: str,
    arch: str,
    model: str,
    fmax: float,
    out_path: str = ".",
    opt_cell: bool = False,
    dispersion: bool = False,
) -> tuple(float, Atoms):
    """Optimize the geometry of a structure."""
    geo = GeomOpt(
        struct=struct,
        device=device,
        arch=arch,
        fmax=fmax,
        calc_kwargs={"model_paths": model, "dispersion": dispersion},
        filter_kwargs={"hydrostatic_strain": opt_cell},
    )
    geo.run()
    if isinstance(struct, Path):
        write(Path(out_path) / f"{struct.stem}-opt.cif", geo.struct)
    return (geo.struct.get_potential_energy(), geo.struct)
