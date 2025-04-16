[![PyPI version][pypi-badge]][pypi-link]
[![Python versions][python-badge]][python-link]
[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]
[![Docs status][docs-badge]][docs-link]

# what is packmm

packmm is a simple python package that allows to build atomistic and molecular
systems which are of interest for materials and molecular modelling.

It tries to generate realistic starting configuration by employing machine learnt
interatomic potential for describing interactions between atoms and Monte Carlo,
Molecular Dynamics and hybrid Monte Carlo.

It is using [janus-core](https://github.com/stfc/janus-core) for MLIPs ([mace mp](https://github.com/ACEsuit/mace-mp) is the default) and for MD and Geometry optimisation
routines.

It provides both a cli and a python api, with some examples below.

## Quick install

```bash
   python3 -m pip install pack-mm   # or
   uv pip install pack-mm

```
or install the latest

```bash

  python3 -m pip install git+https://github.com/ddmms/pack-mm.git # or
  uv pip install git+https://github.com/ddmms/pack-mm.git

```

## Using gpu

by default runs on cpu, if you want to use gpus (recommended for big systems) pass --device cuda to cli or device="cuda" to python
API.

## Jupyter notebook examples


- [Basics](docs/source/tutorials/basic.ipynb) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/pack-mm/blob/main/docs/source/tutorials/basic.ipynb)
- [Surface](docs/source/tutorials/surface.ipynb) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/pack-mm/blob/main/docs/source/tutorials/surface.ipynb)
- [Water](docs/source/tutorials/water.ipynb) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/pack-mm/blob/main/docs/source/tutorials/water.ipynb)
- [Interstitials](docs/source/tutorials/interstitials.ipynb) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/pack-mm/blob/main/docs/source/tutorials/interstitials.ipynb)
- [Mof](docs/source/tutorials/mof.ipynb) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/pack-mm/blob/main/docs/source/tutorials/mof.ipynb)

## CLI examples


### MOF in spherical pocket

```bash

   packmm --system examples/data/UiO-66.cif --molecule H2O --nmols 10  --where sphere --centre 10.0,10.0,10.0 --radius 5.0 --geometry

```

![](examples/pics/UiO66water.webp)

### Zeolite in cylindrical channel


```bash

   packmm --system examples/data/MFI.cif --molecule H2O --nmols 30  --where cylinderY --centre 10.0,10.0,13.0 --radius 3.5 --height 19.00  --no-geometry

```

![](examples/pics/MFIwater.webp)

### NaCl on surface

```bash
   packmm --system examples/data/NaCl.cif --molecule H2O --nmols 30  --where box --centre 8.5,8.5,16.0 --a 16.9 --b 16.9 --c 7.5 --no-geometry

```

![](examples/pics/NaClwater.webp)

### MOF ellipsoid

first add a methanol

```bash

packmm --system examples/data/Cu2L.cif --molecule examples/data/Ethanol.xyz --nmols 1  --where sphere --centre 5.18,8.15,25.25 --radius 1 --model small-0b2 --geometry

```

![](examples/pics/Cu2L-ethanol.webp)

``` bash

packmm --system Cu2L-ethanol.cif --molecule H2O --nmols 10  --where ellipsoid --centre 5.18,8.15,25.25 --a 5.18 --b 8.15 --c 8.25 --no-geometry --model small-0b2


```

![](examples/pics/Cu2l-ethanol-water.webp)

### Liquid water

```bash

packmm --molecule H2O --nmols 33  --where anywhere  --cell-a 10.0 --cell-b 10.0 --cell-c 10.0  --model small-0b2


```

![](examples/pics/water.webp)

### interstitials

```bash

packmm --system Pd-super.cif --molecule H2 --nmols 50  --where anywhere   --model small-0b2

```

before optimisation

![](examples/pics/Pd-super+50H2.webp)


after optimisation

![](examples/pics/Pd-H2.webp)


### full list of options

```bash

  packmm --help

   Usage: packmm [OPTIONS]

 Pack molecules into a system based on the specified parameters.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --system                                 TEXT                        The original box in which   │
│                                                                      you want to add particles.  │
│                                                                      If not provided, an empty   │
│                                                                      box will be created.        │
│                                                                      [default: None]             │
│ --molecule                               TEXT                        Name of the molecule to be  │
│                                                                      processed, ASE-recognizable │
│                                                                      or ASE-readable file.       │
│                                                                      [default: H2O]              │
│ --nmols                                  INTEGER                     Target number of molecules  │
│                                                                      to insert.                  │
│                                                                      [default: -1]               │
│ --ntries                                 INTEGER                     Maximum number of attempts  │
│                                                                      to insert each molecule.    │
│                                                                      [default: 50]               │
│ --seed                                   INTEGER                     Random seed for             │
│                                                                      reproducibility.            │
│                                                                      [default: 2025]             │
│ --where                                  [anywhere|sphere|box|cylin  Where to insert the         │
│                                          derZ|cylinderY|cylinderX|e  molecule. Choices:          │
│                                          llipsoid]                   'anywhere', 'sphere',       │
│                                                                      'box', 'cylinderZ',         │
│                                                                      'cylinderY', 'cylinderX',   │
│                                                                      'ellipsoid'.                │
│                                                                      [default: anywhere]         │
│ --centre                                 TEXT                        Centre of the insertion     │
│                                                                      zone, coordinates in Å,     │
│                                                                      e.g., '5.0, 5.0, 5.0'.      │
│                                                                      [default: None]             │
│ --radius                                 FLOAT                       Radius of the sphere or     │
│                                                                      cylinder in Å, depending on │
│                                                                      the insertion volume.       │
│                                                                      [default: None]             │
│ --height                                 FLOAT                       Height of the cylinder in   │
│                                                                      Å.                          │
│                                                                      [default: None]             │
│ --a                                      FLOAT                       Side of the box or          │
│                                                                      semi-axis of the ellipsoid, │
│                                                                      in Å, depends on the        │
│                                                                      insertion method.           │
│                                                                      [default: None]             │
│ --b                                      FLOAT                       Side of the box or          │
│                                                                      semi-axis of the ellipsoid, │
│                                                                      in Å, depends on the        │
│                                                                      insertion method.           │
│                                                                      [default: None]             │
│ --c                                      FLOAT                       Side of the box or          │
│                                                                      semi-axis of the ellipsoid, │
│                                                                      in Å, depends on the        │
│                                                                      insertion method.           │
│                                                                      [default: None]             │
│ --device                                 TEXT                        Device to run calculations  │
│                                                                      on (e.g., 'cpu' or 'cuda'). │
│                                                                      [default: cpu]              │
│ --model                                  TEXT                        ML model to use.            │
│                                                                      [default: medium-omat-0]    │
│ --arch                                   TEXT                        MLIP architecture to use.   │
│                                                                      [default: mace_mp]          │
│ --temperature                            FLOAT                       Temperature for the Monte   │
│                                                                      Carlo acceptance rule.      │
│                                                                      [default: 300.0]            │
│ --cell-a                                 FLOAT                       Side of the empty box along │
│                                                                      the x-axis in Å.            │
│                                                                      [default: 20.0]             │
│ --cell-b                                 FLOAT                       Side of the empty box along │
│                                                                      the y-axis in Å.            │
│                                                                      [default: 20.0]             │
│ --cell-c                                 FLOAT                       Side of the empty box along │
│                                                                      the z-axis in Å.            │
│                                                                      [default: 20.0]             │
│ --fmax                                   FLOAT                       force tollerance for        │
│                                                                      optimisation if needed.     │
│                                                                      [default: 0.1]              │
│ --geometry              --no-geometry                                Perform geometry            │
│                                                                      optimization at the end.    │
│                                                                      [default: geometry]         │
│ --out-path                               TEXT                        path to save various        │
│                                                                      outputs.                    │
│                                                                      [default: .]                │
│ --install-completion                                                 Install completion for the  │
│                                                                      current shell.              │
│ --show-completion                                                    Show completion for the     │
│                                                                      current shell, to copy it   │
│                                                                      or customize the            │
│                                                                      installation.               │
│ --help                                                               Show this message and exit. │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯


```

## License

[MIT License](LICENSE)

## Funding



[doi-link]: https://zenodo.org/badge/latestdoi/937083576
[doi-badge]: https://zenodo.org/badge/937083576.svg
[python-badge]: https://img.shields.io/pypi/pyversions/pack-mm.svg
[python-link]: https://pypi.org/project/pack-mm/
[ci-badge]: https://github.com/ddmms/pack-mm/actions/workflows/build.yml/badge.svg?branch=main
[ci-link]: https://github.com/ddmms/pack-mm/actions
[cov-badge]: https://coveralls.io/repos/github/ddmms/pack-mm/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/ddmms/pack-mm?branch=main
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-link]: https://opensource.org/license/MIT
[pypi-badge]: https://badge.fury.io/py/pack-mm.svg
[pypi-link]: https://pypi.org/project/pack-mm/
[docs-badge]: https://github.com/ddmms/pack-mm/actions/workflows/python-publish.yml/badge.svg
[docs-link]: https://ddmms.github.io/pack-mm/
