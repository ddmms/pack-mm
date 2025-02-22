"""Command line for packmm."""

# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 22-02-2025
# ©alin m elena,
from __future__ import annotations

from enum import Enum

import typer

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


app = typer.Typer(no_args_is_help=True)


@app.command()
def packmm(
    system: str | None = typer.Option(
        None,
        help="""The original box in which you want to add particles.
        If not provided, an empty box will be created.""",
    ),
    molecule: str = typer.Option(
        "H2O",
        help="""Name of the molecule to be processed, ASE-recognizable or
        ASE-readable file.""",
    ),
    nmols: int = typer.Option(-1, help="Target number of molecules to insert."),
    ntries: int = typer.Option(
        50, help="Maximum number of attempts to insert each molecule."
    ),
    seed: int = typer.Option(2025, help="Random seed for reproducibility."),
    where: InsertionMethod = typer.Option(
        InsertionMethod.ANYWHERE,
        help="""Where to insert the molecule. Choices: 'anywhere', 'sphere',
        'box', 'cylinderZ', 'cylinderY', 'cylinderX', 'ellipsoid'.""",
    ),
    centre: str | None = typer.Option(
        None,
        help="""Centre of the insertion zone in fractional coordinates,
        e.g., '0.12,0.4,0.5'.""",
    ),
    radius: float | None = typer.Option(
        None,
        help="""Radius of the sphere or cylinder in Å,
        depending on the insertion volume.""",
    ),
    height: float | None = typer.Option(
        None, help="Height of the cylinder in fractional coordinates."
    ),
    a: float | None = typer.Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, fractional,
        depends on the insertion method.""",
    ),
    b: float | None = typer.Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, fractional,
        depends on the insertion method.""",
    ),
    c: float | None = typer.Option(
        None,
        help="""Side of the box or semi-axis of the ellipsoid, fractional,
        depends on the insertion method.""",
    ),
    device: str = typer.Option(
        "cpu", help="Device to run calculations on (e.g., 'cpu' or 'cuda')."
    ),
    model: str = typer.Option("medium-omat-0", help="ML model to use."),
    arch: str = typer.Option("mace_mp", help="MLIP architecture to use."),
    temperature: float = typer.Option(
        300.0, help="Temperature for the Monte Carlo acceptance rule."
    ),
    cell_a: float = typer.Option(20.0, help="Side of the empty box along the x-axis."),
    cell_b: float = typer.Option(20.0, help="Side of the empty box along the y-axis."),
    cell_c: float = typer.Option(20.0, help="Side of the empty box along the z-axis."),
    fmax: float = typer.Option(
        0.1, help="force tollerance for optimisation if needed."
    ),
    geometry: bool = typer.Option(
        True, help="Perform geometry optimization at the end."
    ),
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
    print(f"{device=}")
    print(f"{temperature=}")
    print(f"{fmax=}")
    print(f"{geometry=}")
    if nmols == -1:
        print("nothing to do, no molecule to insert")
        raise typer.Exit(0)

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
        geometry=geometry,
        ca=cell_a,
        cb=cell_b,
        cc=cell_c,
    )


if __name__ == "__main__":
    app()
