#!/bin/sh
#SBATCH -J filtering # Job name
#SBATCH -o  ./out/filtering.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A5000 # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:1
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=16
#SBATCH     --mail-user=jihyunlee@postech.ac.kr
#SBATCH     --mail-type=ALL

#cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module purge
module add cuda/10.4
module add cuDNN/cuda/10.4/8.0.4.30
module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA_new "
conda activate QA_new

export PYTHONPATH=.




python learn_teacher_ver.py \
--short 0 \
--only_labeling 1 \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--verify_data_path /home/jihyunlee/two-teacher/pptod/data/multiwoz/data/amt/change_gen_$2.json \
--fine_trained /home/jihyunlee/two-teacher/model/$2/teacher_ver3up_1/model.pt \
--save_prefix /home/jihyunlee/pptod/data/multiwoz/data/pseudo/seed$1/use_lists/ \
--file_name  aug_$2 \
--test_batch_size_per_gpu 16 \
--gpus 1