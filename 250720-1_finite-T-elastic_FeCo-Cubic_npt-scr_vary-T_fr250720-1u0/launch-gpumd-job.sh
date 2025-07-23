#!/bin/bash

# where nep.txt is
## NEP89, its dire: /home/wangy43/codes_inst_loca/nep_pote_lib/nep89_20250409/
nep_dest="/home/wangy43/codes_inst_loca/nep_pote_lib/SongKK-FeCoWRe250720_nep/SongKK-FeCoWRe250720_nep.txt"

## wangyz-2024-C-nep
#nep_dest="/home/wangy43/codes_inst_loca/nep_pote_lib/a-C+biGraphGraphite+glassy+rho0.1-1.5_PBE_NEP3_4.2_3.7_220221-240219forAdancedGPUMD-v3.9.nep"

## newly trained wangyz-2024-C-nep with higher accuracy
#nep_dest="/home/wangy43/codes_inst_loca/nep_pote_lib/highly-accurate-C-nep6738_250311.txt"

## ZhangBH-2025-CHON-nep
#nep_dest="/home/wangy43/codes_inst_loca/nep_pote_lib/ZhangBH-nep_0325_end.txt"
########################################################


#running code and wall time
code="/home/wangy43/codes_inst_loca/GPUMD-v4.0std_250501/src/gpumd"         # newest version
#code="/home/wangy43/codes_inst_loca/GPUMD-v3.9.5lmaster_250311/src/gpumd"
#code="/home/wangy43/codes_inst_loca/GPUMD-v3.9.4std_240601/src/gpumd"
#code="/home/wangy43/codes_inst_loca/GPUMD-v3.9.5lmaster_241127/src/gpumd"
#code="/home/wangy43/codes_inst_loca/GPUMD-v3.9.5lmaster_241102/src/gpumd"
wal_time="01-00:00:00"
###########################################################



# where model.xyz is 
#re_dir_exyz="/home/wangy43/.yasconf/public-structure-models-for-initializing-jobs/241113-demo_create_modelexyz_diaC_density-varySize"
re_dir_exyz="/scratch/work/wangy43/work-collaborator/ChengHX_proj_finite-T-elastic/250720-1u0_model.xyz"


# job and run.in
is=(300 700 1100)                               
#is=(300 700 1100)
js=(1)   
ks=(1)
#$ru_def_step=(2965500 2071000 1561600 1237200 1014900)


cwd=$(pwd)
for i in ${!is[*]}
do
        for j in ${!js[*]}
        do
		    for k in ${!ks[*]} ; do  
			    writ_dire="job_${is[i]}_${js[j]}_${ks[k]}"   #?????????????????????
               	mkdir -p $writ_dire
                cd $writ_dire
#==run.in ==============
                	cat > run.in << !
potential  ${nep_dest}

velocity    ${is[i]}    

ensemble    npt_scr ${is[i]} ${is[i]} 100 0 0 0 0 0 0 100 100 100 100 100 100 1000
time_step   1
dump_thermo  100
dump_exyz       100000 1 1 1         
run       1000000

ensemble    npt_scr ${is[i]} ${is[i]} 100 0 0 0 0 0 0 100 100 100 100 100 100 1000
time_step   1
dump_thermo  100
dump_exyz       100000 1 1 1         
run       1000000





#--- candidate ensembles -----
#ensemble         nvt_bdp/nhc/ber 300 300 100
#ensemble         npt_scr/ber 400 400 100 0 0 0 1000 1000 10 1000


#---- HENMD calc ------------
#ensemble        nvt_nhc 300 300 100
#time_step       1
#dump_thermo     1000
#run             200000

#ensemble        nvt_nhc 300 300 100
#time_step       1
#dump_thermo     1000
#compute_hnemd   1000 0 0 0.0002
#compute_shc     4 250 2 1000 400
#dump_restart    3000000
#run             3000000


#--- PDOS NPC1.5 ----
#PS: of course you can do PDOS calc in other ensembles
#ensemble        nve 
#time_step       1
#run             50000
#ensemble        nve
#time_step       1
#dump_thermo     1000
#compute_dos     5 500 400 num_dos_points 2000   # 1/(2*5 fs)=f=100THz, 2pi*f must be greater than the max of frequecy you set. i.e. 100*2pi>400
#run             500000
!



#^^^ Prepare model.xyz ^^^^^
			    read_dire="${re_dir_exyz}"              # full path
                #read_dire="${cwd}/../${re_dir_exyz}"     # relative path
			    #model="model_C_${is[i]}_${js[j]}.xyz" 		# ?????????????
			    model="333-FeCo-CONTCAR.xyz"
                cp $read_dire/$model ./model.xyz
#^^^^^^^^^^^^^^^^^^^^^^^^^^



#/// Prepare sbatch script ////
                if [[ 1 -ge 10 ]] ; then          # condition ??????????
                        card="ampere|hopper"   # ampere|volta,pascal,or kepler  ???????
                else
                        card="volta|ampere|hopper"
                fi
                case_name=$(pwd |awk -F"/" '{printf "%s/%s", $'$(pwd |awk -F"/" '{print NF-1}')', $'$(pwd |awk -F"/" '{print NF}')'}')
			    cat > submit.sbatch <<!
#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --constraint="$card"     # a feature-specific card like ampere|volta,pascal,or kepler
##SBATCH --exclude=dgx8   # 排除有问题的节点
#SBATCH --time="$wal_time"                       # hh:mm:ss or dd-hh
#SBATCH --job-name="$case_name"
##SBATCH --mem=80G                               # Request CPU memory for GPU task
#SBATCH --mail-type=FAIL   --mail-user=yanzhowang@gmail.com          #BEGIN, END, FAIL, ALL.

module load cuda
srun $code
!
#//////////////////




#!!!! submit job !!!!!
		        sbatch submit.sbatch; sleep 1s
#!!!!!!!!!!!!!!!!


                cd $cwd
               	echo "${is[i]} ${js[j]} ${ks[k]}  done ..."
		    done
        done
done
