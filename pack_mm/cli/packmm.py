"""Command line for packmm."""

# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena,
from __future__ import annotations

from enum import Enum

from janus_core.cli.utils import yaml_converter_callback
from typer import Exit, Option, Typer
from typer_config import use_config

from pack_mm.core.core import pack_molecules


class InsertionMethod(str, Enum):
    """Insertion options."""

    ANYWHERE = "anywhere"
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER_Z = "cylinderZ"
    CYLINDER_Y = "cylinderY"
    CYLINDER_X = "cylinderX"
    ELLIPSOID = "ellipsoid"


class InsertionStrategy(str, Enum):
    """Insertion options."""

    # propose randomly a point
    MC = "mc"
    # hybrid monte carlo
    HMC = "hmc"


class RelaxStrategy(str, Enum):
    """Relaxation options."""

    GEOMETRY_OPTIMISATION = "geometry_optimisation"
    MD = "md"


app = Typer(no_args_is_help=True)


@app.command()
@use_config(yaml_converter_callback)
def packmm(
    system: str | None = Option(
        None,
        help="""The original box in which you want to add particles.
        If not provided, an empty box will be created.""",
    ),
    molecule: str = Option(
        "H2O",
        help="""Name of the molecule to be processed, ASE-recognizable or
        ASE-readable file.""",
    ),
    nmols: int = Option(-1, help="Target number of molecules to insert."),
    ntries: int = Option(
        50, help="Maximum number of attempts to insert each molecule."
    ),
    every: int = Option(
        -1, help="Run MD-NVE or Geometry optimisation everyth insertion."
    ),
    seed: int = Option(2025, help="Random seed for reproducibility."),
    md_steps: int = Option(10, help="Number of steps to run MD."),
    md_timestep: float = Option(1.0, help="Timestep for MD integration, in fs."),
    where: InsertionMethod = Option(
        InsertionMethod.ANYWHERE,
        help="""Where to insert the molecule. Choices: 'anywhere', 'sphere',
        'box', 'cylinderZ', 'cylinderY', 'cylinderX', 'ellipsoid'.""",
    ),
    insert_strategy: InsertionStrategy = Option(
        InsertionStrategy.MC,
        help="""How to insert a new molecule. Choices: 'mc', 'hmc',""",
    ),
    relax_strategy: RelaxStrategy = Option(
        RelaxStrategy.GEOMETRY_OPTIMISATION,
        help="""How to relax the system to get more favourable structures.
            Choices: 'geometry_optimisation', 'md',""",
    ),
    centre: str | None = Option(
        None,
        help="""Centre of the insertion zone, coordinates in Å,
        e.g., '5.0, 5.0, 5.0'.""",
    ),
    radius: float | None = Option(
        None,
        help="""Radius of the sphere or cylinder in Å,
        depending on the insertion volume.""",
    ),
    height: float | None = Option(None, help="Height of the cylinder in Å."),
    a: float | None = Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, in Å,
        depends on the insertion method.""",
    ),
    b: float | None = Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, in Å,
        depends on the insertion method.""",
    ),
    c: float | None = Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, in Å,
        depends on the insertion method.""",
    ),
    device: str = Option(
        "cpu", help="Device to run calculations on (e.g., 'cpu' or 'cuda')."
    ),
    model: str = Option("medium-omat-0", help="ML model to use."),
    dispersion: bool = Option(False, help="Include a dispersion correction."),
    arch: str = Option("mace_mp", help="MLIP architecture to use."),
    temperature: float = Option(
        300.0, help="Temperature for the Monte Carlo acceptance rule."
    ),
    md_temperature: float = Option(
        100.0, help="Temperature for the Molecular dynamics relaxation."
    ),
    cell_a: float = Option(20.0, help="Side of the empty box along the x-axis in Å."),
    cell_b: float = Option(20.0, help="Side of the empty box along the y-axis in Å."),
    cell_c: float = Option(20.0, help="Side of the empty box along the z-axis in Å."),
    fmax: float = Option(0.1, help="force tollerance for optimisation if needed."),
    threshold: float = Option(
        0.92,
        help="""percentage of the single molecule energy above which
                the acceptance energy difference must
                be considered for acceptance.""",
    ),
    geometry: bool = Option(True, help="Perform geometry optimization at the end."),
    out_path: str = Option(".", help="path to save various outputs."),
):
    """Pack molecules into a system based on the specified parameters."""
    print("Script called with following input")
    print(f"{system=}")
    print(f"{nmols=}")
    print(f"{molecule=}")
    print(f"{ntries=}")
    print(f"{seed=}")
    print(f"where={where.value}")
    print(f"{centre=}")
    print(f"{radius=}")
    print(f"{height=}")
    print(f"{a=}")
    print(f"{b=}")
    print(f"{c=}")
    print(f"{cell_a=}")
    print(f"{cell_b=}")
    print(f"{cell_c=}")
    print(f"{arch=}")
    print(f"{model=}")
    print(f"{dispersion=}")
    print(f"{device=}")
    print(f"{temperature=}")
    print(f"{fmax=}")
    print(f"{threshold=}")
    print(f"{geometry=}")
    print(f"{out_path=}")
    print(f"{every=}")
    print(f"insert_strategy={insert_strategy.value}")
    print(f"relax_strategy={relax_strategy.value}")
    print(f"{md_steps=}")
    print(f"{md_timestep=}")
    print(f"{md_temperature=}")
    if nmols == -1:
        print("nothing to do, no molecule to insert")
        raise Exit(0)

    center = centre
    if centre:
        center = tuple(map(float, centre.split(",")))
        lc = [x < 0.0 for x in center]
        if len(center) != 3 or any(lc):
            err = "Invalid centre 3 coordinates expected!"
            print(f"{err}")
            raise Exception("Invalid centre 3 coordinates expected!")

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
        temperature=temperature,
        ntries=ntries,
        fmax=fmax,
        threshold=threshold,
        geometry=geometry,
        cell_a=cell_a,
        cell_b=cell_b,
        cell_c=cell_c,
        out_path=out_path,
        every=every,
        relax_strategy=relax_strategy,
        insert_strategy=insert_strategy,
        md_steps=md_steps,
        md_timestep=md_timestep,
        md_temperature=md_temperature,
    )
