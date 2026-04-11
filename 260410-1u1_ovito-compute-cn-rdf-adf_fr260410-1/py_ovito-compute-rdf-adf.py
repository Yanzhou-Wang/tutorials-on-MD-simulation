#!/usr/bin/env python3

'''
activate ovito env in python beforce run the script
`source ~/venvpy/pyvenv-ovito/bin/activate on my Ubuntu`
'''

r_path="../260410-1_result-restart.xyz_NPC1.2-aC3.2"
job_start_str="job_"
r_f_n="restart.xyz"
rdf_cutoff=10  # angstrom
rdf_bins=int(rdf_cutoff/0.05)

adf_cutoff=2;
adf_bins=180;


import os
from ovito.io import *
from ovito.modifiers import *
import numpy as np

def Proc_rdf(indump, outrdf, rdfcf, bins):
    # Load input data.
    pipeline = import_file(indump)
    nfram = pipeline.source.num_frames

    # Calculate partial RDFs with time average:
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(
            cutoff=rdfcf,
            number_of_bins=bins,
            partial=True))
    pipeline.modifiers.append(
        TimeAveragingModifier(
            operate_on='table:coordination-rdf'))
    TimeAveragingModifier.interval=(nfram//2, nfram-1)

    data = pipeline.compute()
    export_file(pipeline, outrdf, 'txt/table', key='coordination-rdf[average]')

def Proc_adf(indump, outadf, del_atom_type=[], adfcf=[], bins=[]):
    pipeline = import_file(indump)

    data = pipeline.compute()

    # if you want delete some elements before compute adf.
    if del_atom_type != []:
        for di in del_atom_type:
            pipeline.modifiers.append(
                SelectTypeModifier(
                    operate_on='particles',
                    property="Particle Type",
                    types={di}))
        pipeline.modifiers.append(
            DeleteSelectedModifier())
        data = pipeline.compute()

    # Before calculating the RDF, you need to form a key information
    #   and there are several different ways to generate it

    # mode style ref from:
    # https://www.ovito.org/docs/current/python/modules/ovito_modifiers.html#ovito.modifiers.CreateBondsModifier.mode
    if adfcf == []:                      # mode = "VdWRadius"
        # uses a distance cutoff that is derived from the vdw_radius
        pipeline.modifiers.append(CreateBondsModifier(
                mode=CreateBondsModifier.Mode.VdWRadius,lower_cutoff=0.1))
    elif isinstance(adfcf, (int,float)): # mode = "Uniform"
        # uses a single uniform cutoff distance for creating bonds
        cfs = adfcf
        pipeline.modifiers.append(CreateBondsModifier(
                mode=CreateBondsModifier.Mode.Uniform,cutoff=cfs))
    elif isinstance(adfcf, dict):        # mode = "Pairwise"
        # specify a separate cutoff distance for each pairwise combination of particle types
        cbm = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise,lower_cutoff=0.1)
        for di in adfcf:
            p1, p2 = di.split('-')
            cbm.set_pairwise_cutoff(p1, p2, adfcf[di])
            pipeline.modifiers.append(cbm)
    else:
        raise "error with adfcf types, there are three types:\n num, [], dict"

    # Calculate instantaneous bond angle distribution.
    pipeline.modifiers.append(BondAnalysisModifier(bins = bins))
    # Perform time averaging of the DataTable 'bond-angle-distr'.
    pipeline.modifiers.append(TimeAveragingModifier(operate_on='table:bond-angle-distr'))
    # Compute and export the time-averaged histogram to a text file.
    export_file(pipeline, outadf, 'txt/table', key='bond-angle-distr[average]')


def main():
    dirs=[dir for dir in os.listdir(r_path) if dir.startswith(job_start_str) and os.path.isdir(os.path.join(r_path, dir))]
    for dir_name in dirs:
        w_dir=dir_name
        os.makedirs(w_dir, exist_ok=True)      
        r_file=r_f_n
        r_dest_file=os.path.join(r_path, dir_name, r_file)
        indump=r_dest_file
        
        w_rdf_file="rdf.txt"
        w_rdf_dest_file=os.path.join(w_dir, w_rdf_file)
        outrdf=w_rdf_dest_file
        Proc_rdf(indump, outrdf, rdfcf=rdf_cutoff, bins=rdf_bins)
        
        
        w_adf_file="adf.txt"
        w_adf_dest_file=os.path.join(w_dir, w_adf_file)
        outadf=w_adf_dest_file
        Proc_adf(indump, outadf, adfcf=adf_cutoff, bins=adf_bins)
        
          
        print(f"{dir_name} is done ...")


if __name__ == '__main__':
    main()

