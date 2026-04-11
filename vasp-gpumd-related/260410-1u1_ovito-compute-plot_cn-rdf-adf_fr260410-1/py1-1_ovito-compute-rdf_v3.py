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

rdf_cutoff = 5   # angstrom
rdf_bin_width = 0.05   # angstrom
rdf_bins = int(rdf_cutoff / rdf_bin_width)

w_rdf_f_n = "result-rdf.txt"

# time averaging range:
# use the second half of all frames: [nframe//2, nframe-1]
average_from_half_trajectory = True
# ============================================================


import os
from ovito.io import import_file, export_file
from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier


def Proc_rdf(indump, outrdf, rdfcf, bins):
    # Load input data.
    pipeline = import_file(indump)
    nfram = pipeline.source.num_frames

    # Calculate partial RDF.
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(
            cutoff=rdfcf,
            number_of_bins=bins,
            partial=True
        )
    )

    # Time average the RDF table.
    avg_modifier = TimeAveragingModifier(operate_on='table:coordination-rdf')

    if average_from_half_trajectory:
        avg_modifier.interval = (nfram // 2, nfram - 1)

    pipeline.modifiers.append(avg_modifier)

    # Export the time-averaged partial RDF.
    export_file(
        pipeline,
        outrdf,
        'txt/table',
        key='coordination-rdf[average]'
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
        w_rdf_dest_file = os.path.join(w_dir, w_rdf_f_n)

        Proc_rdf(
            indump=r_dest_file,
            outrdf=w_rdf_dest_file,
            rdfcf=rdf_cutoff,
            bins=rdf_bins
        )

        print(f"{dir_name} is done ...")


if __name__ == '__main__':
    main()
    
    
    