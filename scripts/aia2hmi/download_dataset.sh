#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=6:00:00

#SBATCH --qos=preemptible

############################################

year=$1
ouput_dir=$2
redo_all=$3

echo "year: $year , output_dir: $ouput_dir , redo_all: $redo_all"

python ./scripts/aia2hmi/Get_aia2hmi_dataset_4hours.py --year $year --output_dir $ouput_dir --redo_all $redo_all
