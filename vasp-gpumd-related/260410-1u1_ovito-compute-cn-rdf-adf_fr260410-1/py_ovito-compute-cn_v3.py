#!/usr/bin/env python3

'''
activate ovito env in python before run the script
`source ~/venvpy/pyvenv-ovito/bin/activate on my Ubuntu`
'''

r_jb_p_dir = "../260410-1_result-restart.xyz_NPC1.2-aC3.2"
job_start_str = "job_"
r_f_n = "restart.xyz"

# ------------------------------------------------------------------
# Pairwise cutoff dictionary:
# Define cutoff for each element pair explicitly.
# Unspecified pairs will have cutoff = 0.0, i.e. no bond will be created.
#
# Example for pure carbon / pure hydrogen / C-H mixed system:
pair_cutoffs = {
    ("C", "C"): 1.9,
    ("C", "H"): 1.6,
    ("H", "H"): 0.90,
}

# Example for Li-Ni system:
# pair_cutoffs = {
#     ("Li", "Li"): 3.20,
#     ("Li", "Ni"): 2.80,
#     ("Ni", "Ni"): 2.60,
# }

# Optional global lower cutoff:
lower_cutoff = 0.0
# ------------------------------------------------------------------


import os
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier
import numpy as np


def normalize_pair_cutoffs(pair_cutoffs):
    """
    Normalize pair_cutoffs so that ('C','H') and ('H','C') are treated as the same pair.
    If both are given, the later one will overwrite the former.
    """
    normalized = {}
    for pair, cutoff in pair_cutoffs.items():
        if len(pair) != 2:
            raise ValueError(f"Invalid pair key {pair}. Each key must be a 2-tuple like ('C','H').")
        a, b = pair
        key = tuple(sorted((str(a), str(b))))
        normalized[key] = float(cutoff)
    return normalized


def build_type_mapping(data):
    """
    Build:
      1) type-id -> type-name mapping string for header comment
      2) available type-name set for validation/filtering
    """
    type_mapping_list = []
    available_type_names = set()

    type_prop = data.particles.particle_types
    for t in type_prop.types:
        type_mapping_list.append((t.id, t.name))
        available_type_names.add(str(t.name))

    type_mapping_list.sort(key=lambda x: x[0])
    type_mapping_str = ", ".join([f"{tid}={name}" for tid, name in type_mapping_list])

    return type_mapping_str, available_type_names


def filter_pair_cutoffs_for_current_system(normalized_pair_cutoffs, available_type_names):
    """
    Keep only the pairs for which both element/type names exist in the current system.
    Example:
      available_type_names = {'C'}  -> keep only ('C','C')
      available_type_names = {'H'}  -> keep only ('H','H')
      available_type_names = {'C','H'} -> keep ('C','C'), ('C','H'), ('H','H')
    """
    filtered = {}
    skipped = {}

    for (a, b), cutoff in normalized_pair_cutoffs.items():
        if a in available_type_names and b in available_type_names:
            filtered[(a, b)] = cutoff
        else:
            skipped[(a, b)] = cutoff

    return filtered, skipped


def build_pair_cutoff_comment(pair_cutoffs_dict):
    """
    Build a readable comment string for pair cutoffs.
    """
    if not pair_cutoffs_dict:
        return "None"

    items = sorted(pair_cutoffs_dict.items(), key=lambda x: (x[0][0], x[0][1]))
    return ", ".join([f"{a}-{b}={cutoff:.6f}" for (a, b), cutoff in items])


def compute_coordination_from_bonds(topology, num_particles):
    """
    Count coordination number from bond topology.
    topology: Nx2 array, each row is a bonded pair (i, j)
    """
    cn = np.zeros(num_particles, dtype=int)
    for i, j in topology:
        cn[i] += 1
        cn[j] += 1
    return cn


def Proc_cn(indump, outcn, pair_cutoffs, lower_cutoff=0.0):
    # Load input data.
    pipeline = import_file(indump)

    # First compute once to inspect particle types in the input system.
    data_in = pipeline.compute()

    # Build type mapping and available type names.
    type_mapping_str, available_type_names = build_type_mapping(data_in)

    # Normalize user-defined pair cutoff dictionary.
    normalized_pair_cutoffs = normalize_pair_cutoffs(pair_cutoffs)

    # Keep only pairs relevant to the current system.
    active_pair_cutoffs, skipped_pair_cutoffs = filter_pair_cutoffs_for_current_system(
        normalized_pair_cutoffs,
        available_type_names
    )

    # Create bonds using pairwise cutoffs.
    bond_modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    bond_modifier.lower_cutoff = lower_cutoff

    for (a, b), cutoff in active_pair_cutoffs.items():
        bond_modifier.set_pairwise_cutoff(a, b, cutoff)

    pipeline.modifiers.append(bond_modifier)

    # Compute pipeline again after bond creation.
    data = pipeline.compute()

    # Extract particle properties.
    positions = data.particles["Position"]
    particle_types = data.particles["Particle Type"]
    num_particles = data.particles.count

    # Extract bond topology and count CN.
    if data.particles.bonds is not None and data.particles.bonds.count > 0:
        topology = data.particles.bonds.topology
        coord_numbers = compute_coordination_from_bonds(topology, num_particles)
    else:
        coord_numbers = np.zeros(num_particles, dtype=int)

    # Build cutoff comment strings.
    active_pair_cutoff_comment = build_pair_cutoff_comment(active_pair_cutoffs)
    skipped_pair_cutoff_comment = build_pair_cutoff_comment(skipped_pair_cutoffs)

    # Write output file.
    with open(outcn, "w") as f:
        f.write("# particle_index particle_type x y z coordination\n")
        f.write(f"# type mapping: {type_mapping_str}\n")
        f.write(f"# active pair cutoffs: {active_pair_cutoff_comment}\n")
        f.write(f"# skipped pair cutoffs: {skipped_pair_cutoff_comment}\n")

        for i, (ptype, pos, cn) in enumerate(zip(particle_types, positions, coord_numbers)):
            x, y, z = pos
            f.write(f"{i}  {int(ptype)}  {x:.8f}  {y:.8f}  {z:.8f}  {int(cn)}\n")


def main():
    dirs = [
        dir_name for dir_name in os.listdir(r_jb_p_dir)
        if dir_name.startswith(job_start_str)
        and os.path.isdir(os.path.join(r_jb_p_dir, dir_name))
    ]

    for dir_name in dirs:
        w_dir = dir_name
        os.makedirs(w_dir, exist_ok=True)

        r_dest_file = os.path.join(r_jb_p_dir, dir_name, r_f_n)
        w_dest_file = os.path.join(w_dir, "result-cn.txt")

        Proc_cn(
            indump=r_dest_file,
            outcn=w_dest_file,
            pair_cutoffs=pair_cutoffs,
            lower_cutoff=lower_cutoff
        )

        print(f"{dir_name} is done ...")


if __name__ == "__main__":
    main()