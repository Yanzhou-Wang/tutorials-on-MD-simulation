#!/usr/bin/env python3

'''
activate ovito env in python before run the script
`source ~/venvpy/pyvenv-ovito/bin/activate on my Ubuntu`
'''

# ============================================================
# User-defined parameters
# ============================================================
r_jb_p_dir = "../260410-1_result-restart.xyz_NPC1.2-aC3.2"
job_start_str = "job_"
r_f_n = "restart.xyz"

# Pairwise cutoff dictionary for bond creation
# Unspecified pairs will not be used.
pair_cutoffs = {
    ("C", "C"): 1.90,
    ("C", "H"): 1.60,
    ("H", "H"): 0.90,
}

# Optional global lower cutoff for bond creation
lower_cutoff = 0.0

# Bond-length distribution settings
# If bld_cutoff is None, the script will automatically use
# the maximum active pair cutoff in the current system.
bld_cutoff = None
bld_bin_width = 0.05   # angstrom
w_bld_f_n = "result-bld.txt"

# Bond-angle distribution settings
adf_bins = 180
w_adf_f_n = "result-adf.txt"

# Time averaging:
# use the second half of the trajectory: [nframe//2, nframe-1]
average_from_half_trajectory = True
# ============================================================


import os
from ovito.io import import_file, export_file
from ovito.modifiers import CreateBondsModifier, BondAnalysisModifier, TimeAveragingModifier


def normalize_pair_cutoffs(pair_cutoffs):
    """
    Normalize pair_cutoffs so that ('C','H') and ('H','C') are treated as the same pair.
    If both are given, the later one overwrites the former.
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
      1) type-id -> type-name mapping string
      2) available type-name set
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
    Keep only the pairs for which both type names exist in the current system.
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


def prepend_comments_to_file(filename, comment_lines):
    """
    Prepend comment lines to an existing exported text file.
    """
    with open(filename, "r") as f:
        old_content = f.read()

    with open(filename, "w") as f:
        for line in comment_lines:
            f.write(line.rstrip() + "\n")
        f.write(old_content)


def write_empty_distribution_file(filename, comment_lines):
    """
    Write a placeholder file when no active pair cutoffs are available.
    """
    with open(filename, "w") as f:
        for line in comment_lines:
            f.write(line.rstrip() + "\n")
        f.write("# No active pair cutoffs are available for the current system.\n")


def build_pairwise_bond_pipeline(indump, pair_cutoffs, lower_cutoff=0.0):
    """
    Build an OVITO pipeline with pairwise bond creation.
    Automatically keeps only relevant element pairs for the current system.
    """
    pipeline = import_file(indump)

    # Compute once to inspect particle types in the current system
    data_in = pipeline.compute()
    type_mapping_str, available_type_names = build_type_mapping(data_in)

    normalized_pair_cutoffs = normalize_pair_cutoffs(pair_cutoffs)
    active_pair_cutoffs, skipped_pair_cutoffs = filter_pair_cutoffs_for_current_system(
        normalized_pair_cutoffs,
        available_type_names
    )

    # Build pairwise bonds
    bond_modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    bond_modifier.lower_cutoff = lower_cutoff

    for (a, b), cutoff in active_pair_cutoffs.items():
        bond_modifier.set_pairwise_cutoff(a, b, cutoff)

    pipeline.modifiers.append(bond_modifier)

    return pipeline, type_mapping_str, active_pair_cutoffs, skipped_pair_cutoffs


def Proc_bld(indump, outbld, pair_cutoffs, lower_cutoff=0.0, bld_cutoff=None, bld_bin_width=0.05):
    pipeline, type_mapping_str, active_pair_cutoffs, skipped_pair_cutoffs = build_pairwise_bond_pipeline(
        indump=indump,
        pair_cutoffs=pair_cutoffs,
        lower_cutoff=lower_cutoff
    )

    active_pair_cutoff_comment = build_pair_cutoff_comment(active_pair_cutoffs)
    skipped_pair_cutoff_comment = build_pair_cutoff_comment(skipped_pair_cutoffs)

    comment_lines = [
        "# bond length distribution",
        f"# type mapping: {type_mapping_str}",
        f"# active pair cutoffs: {active_pair_cutoff_comment}",
        f"# skipped pair cutoffs: {skipped_pair_cutoff_comment}",
    ]

    if not active_pair_cutoffs:
        write_empty_distribution_file(outbld, comment_lines)
        return

    # Determine length cutoff for bond-length histogram
    if bld_cutoff is None:
        effective_bld_cutoff = max(active_pair_cutoffs.values())
    else:
        effective_bld_cutoff = float(bld_cutoff)

    bld_bins = max(1, int(effective_bld_cutoff / bld_bin_width))

    # Compute bond-length distribution, partitioned by particle type
    pipeline.modifiers.append(
        BondAnalysisModifier(
            bins=bld_bins,
            length_cutoff=effective_bld_cutoff,
            partition=BondAnalysisModifier.Partition.ByParticleType
        )
    )

    nfram = pipeline.source.num_frames
    avg_modifier = TimeAveragingModifier(operate_on='table:bond-length-distr')
    if average_from_half_trajectory:
        avg_modifier.interval = (nfram // 2, nfram - 1)
    pipeline.modifiers.append(avg_modifier)

    export_file(
        pipeline,
        outbld,
        'txt/table',
        key='bond-length-distr[average]'
    )

    prepend_comments_to_file(
        outbld,
        comment_lines + [
            f"# bond length cutoff used for histogram: {effective_bld_cutoff:.6f}",
            f"# number of bins: {bld_bins}",
        ]
    )


def Proc_adf(indump, outadf, pair_cutoffs, lower_cutoff=0.0, adf_bins=180):
    pipeline, type_mapping_str, active_pair_cutoffs, skipped_pair_cutoffs = build_pairwise_bond_pipeline(
        indump=indump,
        pair_cutoffs=pair_cutoffs,
        lower_cutoff=lower_cutoff
    )

    active_pair_cutoff_comment = build_pair_cutoff_comment(active_pair_cutoffs)
    skipped_pair_cutoff_comment = build_pair_cutoff_comment(skipped_pair_cutoffs)

    comment_lines = [
        "# bond angle distribution",
        "# angle unit: degree",
        f"# type mapping: {type_mapping_str}",
        f"# active pair cutoffs: {active_pair_cutoff_comment}",
        f"# skipped pair cutoffs: {skipped_pair_cutoff_comment}",
    ]

    if not active_pair_cutoffs:
        write_empty_distribution_file(outadf, comment_lines)
        return

    # Compute bond-angle distribution, partitioned by particle type
    pipeline.modifiers.append(
        BondAnalysisModifier(
            bins=adf_bins,
            partition=BondAnalysisModifier.Partition.ByParticleType
        )
    )

    nfram = pipeline.source.num_frames
    avg_modifier = TimeAveragingModifier(operate_on='table:bond-angle-distr')
    if average_from_half_trajectory:
        avg_modifier.interval = (nfram // 2, nfram - 1)
    pipeline.modifiers.append(avg_modifier)

    export_file(
        pipeline,
        outadf,
        'txt/table',
        key='bond-angle-distr[average]'
    )

    prepend_comments_to_file(
        outadf,
        comment_lines + [
            f"# number of bins: {adf_bins}",
        ]
    )


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
        w_bld_dest_file = os.path.join(w_dir, w_bld_f_n)
        w_adf_dest_file = os.path.join(w_dir, w_adf_f_n)

        Proc_bld(
            indump=r_dest_file,
            outbld=w_bld_dest_file,
            pair_cutoffs=pair_cutoffs,
            lower_cutoff=lower_cutoff,
            bld_cutoff=bld_cutoff,
            bld_bin_width=bld_bin_width
        )

        Proc_adf(
            indump=r_dest_file,
            outadf=w_adf_dest_file,
            pair_cutoffs=pair_cutoffs,
            lower_cutoff=lower_cutoff,
            adf_bins=adf_bins
        )

        print(f"{dir_name} is done ...")


if __name__ == '__main__':
    main()