#!/bin/bash

cwd=`pwd`
r_dir="$cwd/../260229-1_*"

parse_job_dirs()
{
    # This function parses job directories under $r_dir/job_*.
    #
    # Supported naming rules:
    #   job_xxx_m
    #   job_xxx_m_n
    #
    # Returned global arrays:
    #   is : unique xxx list
    #   js : unique m list
    #   ks : unique n list, only meaningful when job_style=4
    #
    # Returned global variable:
    #   job_style = 3 or 4

    local job_paths=()
    local jp
    local bn
    local nf
    local p1 p2 p3 p4
    local style_ref=""

    is=()
    js=()
    ks=()
    job_style=""

    shopt -s nullglob
    job_paths=($r_dir/job_*)
    shopt -u nullglob

    if [ ${#job_paths[@]} -eq 0 ]; then
        echo "[ERROR] No job directories found under: $r_dir/job_*"
        exit 1
    fi

    for jp in "${job_paths[@]}"
    do
        [ -d "$jp" ] || continue

        bn=$(basename "$jp")

        IFS="_" read -r p1 p2 p3 p4 extra <<< "$bn"

        # Count fields separated by "_"
        nf=$(awk -F"_" '{print NF}' <<< "$bn")

        if [ "$p1" != "job" ]; then
            echo "[ERROR] Invalid job directory name: $bn"
            echo "        Expected: job_xxx_m or job_xxx_m_n"
            exit 1
        fi

        if [ "$nf" -ne 3 ] && [ "$nf" -ne 4 ]; then
            echo "[ERROR] Invalid job directory name: $bn"
            echo "        Expected: job_xxx_m or job_xxx_m_n"
            echo "        The job name should contain 3 or 4 fields separated by '_'."
            exit 1
        fi

        if [ -z "$p2" ] || [ -z "$p3" ]; then
            echo "[ERROR] Invalid job directory name: $bn"
            echo "        Empty field is not allowed."
            exit 1
        fi

        if [ "$nf" -eq 4 ] && [ -z "$p4" ]; then
            echo "[ERROR] Invalid job directory name: $bn"
            echo "        Empty fourth field is not allowed."
            exit 1
        fi

        # m and n should be numbers, integer or decimal.
        if ! [[ "$p3" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "[ERROR] Invalid job directory name: $bn"
            echo "        The third field should be a non-negative number."
            exit 1
        fi

        if [ "$nf" -eq 4 ]; then
            if ! [[ "$p4" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                echo "[ERROR] Invalid job directory name: $bn"
                echo "        The fourth field should be a non-negative number."
                exit 1
            fi
        fi

        # Do not allow mixed job_xxx_m and job_xxx_m_n styles.
        if [ -z "$style_ref" ]; then
            style_ref="$nf"
        else
            if [ "$nf" -ne "$style_ref" ]; then
                echo "[ERROR] The job directory names to be parsed do not follow one consistent naming rule."
                echo "        Mixed job_xxx_m and job_xxx_m_n styles were found."
                echo "        Problematic job directory: $bn"
                exit 1
            fi
        fi
    done

    job_style="$style_ref"

    if [ "$job_style" -eq 3 ]; then
        is=($(ls -d $r_dir/job_* | xargs -n 1 basename | awk -F"_" '{print $2}' | sort -u))
        js=($(ls -d $r_dir/job_* | xargs -n 1 basename | awk -F"_" '{print $3}' | sort -n -u))
        ks=()
    elif [ "$job_style" -eq 4 ]; then
        is=($(ls -d $r_dir/job_* | xargs -n 1 basename | awk -F"_" '{print $2}' | sort -u))
        js=($(ls -d $r_dir/job_* | xargs -n 1 basename | awk -F"_" '{print $3}' | sort -n -u))
        ks=($(ls -d $r_dir/job_* | xargs -n 1 basename | awk -F"_" '{print $4}' | sort -n -u))
    else
        echo "[ERROR] Unknown job_style: $job_style"
        exit 1
    fi

    echo "[INFO] job_style = $job_style"
    echo "[INFO] is = ${is[*]}"
    echo "[INFO] js = ${js[*]}"
    if [ "$job_style" -eq 4 ]; then
        echo "[INFO] ks = ${ks[*]}"
    fi
}

parse_job_dirs


for i in ${!is[*]}
do
    wd="job_${is[i]}";
	rm -rf $wd
	mkdir $wd
	for j in ${!js[*]}
	do
		
		wf="result-energy-vs-strain_${js[j]}_DFT.txt"
		for k in ${!ks[*]}
		do
			strain=${ks[k]}
			jn="job_${is[i]}_${js[j]}_${ks[k]}"
			n_sys=$(grep "number of ions" $r_dir/$jn/OUTCAR |awk '{print $12}' | tail -n 1)
			e_p=$(grep "free  energy   TOTEN" $r_dir/$jn/OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 / '$n_sys'}') 
			echo "$strain	$e_p" | tee -a $wd/$wf
		done
		sed -i '1i#Strain	Energy(eV/atom)' $wd/$wf
		echo ">>>>>>>>>>>>>>>> ${is[i]} | ${js[j]}.... <<<<<<<<<<<<<<<<<<<<<"
	done
done
