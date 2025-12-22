#!/usr/bin/env python3
"""
Generate supercell model from a SiC-like unit cell (here only C sublattice)
using ASE, translated from the original MATLAB script.

Author (original MATLAB): yanzhowang@gmail.com
"""

import numpy as np
from ase import Atoms
from ase.io import write

# --- Unit cell basis (fractional coordinates) ---
# Corresponds to r0_1 in the MATLAB script
r0_1 = np.array([
    [0.0,   0.0,   0.0],
    [0.0,   0.5,   0.5],
    [0.5,   0.0,   0.5],
    [0.5,   0.5,   0.0],
    [0.25,  0.25,  0.25],
    [0.25,  0.75,  0.75],
    [0.75,  0.25,  0.75],
    [0.75,  0.75,  0.25],
])

# If later you want to add another species (e.g. Si), you can define r0_2 here
# and add it to species_and_basis (see below).

# --- Basic constants (same as MATLAB) ---
atom_weight = 12.0              # atomic weight of C
N_unit_cell_atom = len(r0_1)    # number of atoms in the unit cell
atom_mass_unit = 1.6605e-24     # g
unit_mass = N_unit_cell_atom * atom_mass_unit * atom_weight  # g

# List of densities (g/cm^3)
densities = [1.5, 3.25]

# PBC flags: 'T T T' in MATLAB
pbc_flags = [True, True, True]

# Element labels
ele1 = "C"
name_label = ele1

# List of (symbol, basis) pairs, keeping the structure extensible
species_and_basis = [
    (ele1, r0_1),
    # e.g. later you can do: ("Si", r0_2),
]

# Replications along a, b, c directions (nabc in MATLAB)
nabc = np.array([3, 3, 3], dtype=int)

for rho in densities:
    # cubic unit cell length (Å), based on mass and density
    unit_box_length = (unit_mass / rho) ** (1.0 / 3.0) * 1.0e8  # Å

    # Orthorhombic unit cell (diagonal)
    # latt0 in MATLAB is diag(unit_box_length, unit_box_length, unit_box_length)
    # We only need the diagonal values here:
    cell_lengths = nabc * unit_box_length  # supercell lengths along x,y,z (Å)
    cell = np.diag(cell_lengths)

    positions = []
    symbols = []

    # Build the supercell (loops follow MATLAB structure)
    for symbol, basis in species_and_basis:
        # na, nb, nc correspond to a, b, c replica indices
        for na in range(nabc[0]):
            for nb in range(nabc[1]):
                for nc in range(nabc[2]):
                    shift = np.array([na, nb, nc], dtype=float)
                    for frac in basis:
                        # MATLAB: ([na, nb, nc] + r0) .* [latt0(1,1), latt0(2,2), latt0(3,3)]
                        # Here latt0(ii,ii) = unit_box_length, so:
                        pos = (shift + frac) * unit_box_length  # Cartesian Å
                        positions.append(pos)
                        symbols.append(symbol)

    positions = np.array(positions)
    atoms = Atoms(symbols, positions=positions)
    atoms.set_cell(cell)
    atoms.set_pbc(pbc_flags)

    model_name = f"model_{name_label}_rho{rho:g}_size{len(atoms)}.xyz"
    write(model_name, atoms, format="extxyz")

    print(f"{model_name} done ...")
