#!/bin/bash

cwd=`pwd`

rd="$cwd/../0optimize"
code="$cwd/py_identify-vasp-stru_2_prim-conv-orth.py"


for i in `ls -d $rd/id-* |awk -F"/" '{print $NF}'`
do
    jn=$i
    mkdir -p $jn
    cd $jn
    $code $rd/$i/CONTCAR
    cd $cwd

    echo ">>> $i done ..."
    
done

