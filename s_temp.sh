# !/bin/bash
for var in 1
do
sbatch s_run_teacher_ver_up.sh $var 0.3
done

# for var in 0.05 0.5 0.25 0.01
# do
# sbatch s_run_teacher_ver_labeling.sh 1 $var
# done