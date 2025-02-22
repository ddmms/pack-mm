Sphere
------


insert 2 water molecules in a Metal organic framework pore

.. code:: bash

   packmm --system examples/data/UiO-66.cif --molecule H2O --nmols 10  --where sphere --centre 10.0,10.0,10.0 --radius 5.0 --geometry

.. code:: bash

   packmm --system examples/data/MFI.cif --molecule H2O --nmols 30  --where cylinderY --centre 10.0,10.0,13.0 --radius 3.5 --height 19.00  --no-geometry


.. code:: bash

   packmm --system examples/data/NaCl.cif --molecule H2O --nmols 30  --where box --centre 8.5,8.5,16.0 --a 16.9 --b 16.9 --c 7.5 --no-geometry
