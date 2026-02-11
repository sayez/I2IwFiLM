#!/bin/bash

batch_size=16

src_dtype=("whitelight" )
tgt_dtype=("calcium" )

f_factor=1

exp="i2iwfilm_S2"
#  -> ${oc.env:I2IWFILM_PATH}/outputs/wl2cal/S1/f1_I2IwFiLM_S1_seed_0 <--- test run (MUST BE ABSOLUTE PATH)

s1_run_dir=$1
s1_model_path=$2
mlp_hidden=$3

# get the 4 first characters of the mlp_hidden
hidden_size=${mlp_hidden:1:3}

l1_weight=$4
fft_l1_weight=$5
ssim_weight=$6
l2_weight=$7
diff_l1_weight=$8

max_epochs=600
shuffle=$9

variable_lr=${10}

seed=0

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "The directory of the script is: $DIR"
i2iwfilm_path="$(dirname "$DIR")/.."
echo "I2IwFiLM path set to: $i2iwfilm_path"

job_ids=()


for i in "${!src_dtype[@]}"; do

    
    src=${src_dtype[$i]}
    tgt=${tgt_dtype[$i]}

    job_name="I2IwFiLM_S2_${src}2${tgt}"

    echo "job name: $job_name"
    echo "sbatch ./scripts/wl2cal/i2iwfilm_S2_train.sh $job_name $src $tgt $i2iwfilm_path $exp $max_epochs $batch_size $shuffle $variable_lr $s1_run_dir $s1_model_path $l1_weight $fft_l1_weight $ssim_weight $l2_weight $diff_l1_weight $mlp_hidden $seed "
    job_id=$(sbatch ./scripts/wl2cal/i2iwfilm_S2_train.sh $job_name $src $tgt $i2iwfilm_path $exp $max_epochs $batch_size $shuffle $variable_lr $s1_run_dir $s1_model_path $l1_weight $fft_l1_weight $ssim_weight $l2_weight $diff_l1_weight $mlp_hidden $seed  | grep -o -E '[0-9]+')
    
    job_id=$(echo $job_id | grep -o -E '[0-9]+')
    job_ids+=($job_id)

done

echo "job ids: ${job_ids[@]}"