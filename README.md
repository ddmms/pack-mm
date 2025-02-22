# pack materials and molecules

[![Python versions][python-badge]][python-link]
[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![License][license-badge]][license-link]

## Quick install


## CLI examples


### MOF in spherical pocket

```bash

   packmm --system examples/data/UiO-66.cif --molecule H2O --nmols 10  --where sphere --centre 10.0,10.0,10.0 --radius 5.0 --geometry

```

### Zeolite in cylindrical channel


```bash

   packmm --system examples/data/MFI.cif --molecule H2O --nmols 30  --where cylinderY --centre 10.0,10.0,13.0 --radius 3.5 --height 19.00  --no-geometry

```

### NaCl on surface

```bash
   packmm --system examples/data/NaCl.cif --molecule H2O --nmols 30  --where box --centre 8.5,8.5,16.0 --a 16.9 --b 16.9 --c 7.5 --no-geometry

```

### MOF ellipsoid


[python-badge]: https://img.shields.io/pypi/pyversions/pack-mm.svg
[python-link]: https://pypi.org/project/pack-mm/
[ci-badge]: https://github.com/ddmms/pack-mm/actions/workflows/build.yml/badge.svg?branch=main
[ci-link]: https://github.com/ddmms/pack-mm/actions
[cov-badge]: https://coveralls.io/repos/github/ddmms/pack-mm/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/ddmms/pack-mm?branch=main
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-link]: https://opensource.org/license/MIT
