#!/bin/bash

#1. 写在前面的话：脚本执行前，要确保把异常的作业剔除掉, 因为本脚本不负责检查异常作业，只会认为用户完成的所有作业都是健康结束的。如果异常作业没被剔除，改脚本执行的结果可能会病态。
#
#2. Usage: ./script.sh 该脚本从r_dir="../260225-2_*"读入算例的父目录，用户把父目录下的算例文件夹列表放入到一维数组job_items=()即可。
#
#3. 因此，用户需要指定的INPUTS，只有这两点：
#     1) r_dir="算例文件夹的父目录"
#     2）job_items=(算例文件夹名字)
#
#4. OUTPUT1: "pro-elastic.log" 是elastic过程数据日志
#   OUTPUT2: "result-tabled-elastic.txt" 是已整理好的表格化的弹性常数文本数据


r_dir="../260224-2_*"
w_f1="proc-elastic.log"
w_f2="result-tabled-elastic.txt"

rm -rf $w_f1 $w_f2

job_items=(
job_Li-mp-604313_0
)

for i in ${job_items[*]}
do
     #echo "$i" | tee -a $w_f1
     echo "$i" >> $w_f1
	 grep -A 8 "TOTAL ELASTIC MODULI" $r_dir/$i/OUTCAR | tail -n 6 | awk '{printf "%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n", $2/10, $3/10, $4/10, $5/10, $6/10, $7/10}' >> $w_f1
done


r_f="$w_f1"

echo "#job-id	C11	C12	C13	C22	C23	C33	C44	C55	C66 (GPa)" >> $w_f2

for i in ${job_items[*]}
do
	c11=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $1}')
	c12=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $2}')
	c13=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '1p' |awk '{print $3}')
	c22=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '2p' |awk '{print $2}')
	c23=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '2p' |awk '{print $3}')
	c33=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '3p' |awk '{print $3}')
	c44=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '4p' |awk '{print $4}')
	c55=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '5p' |awk '{print $5}')
	c66=$(grep -A 6 $i $r_f |tail -n 6 | sed -n '6p' |awk '{print $6}')
	echo "$i  $c11  $c12  $c13  $c22  $c23  $c33  $c44  $c55  $c66" |tee -a tem.tem
done
column -t tem.tem >> $w_f2
rm -f tem.tem
