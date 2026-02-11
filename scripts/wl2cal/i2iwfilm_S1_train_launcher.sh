#!/bin/bash

batch_size=16

src_dtype=("whitelight" )
tgt_dtype=("calcium" )

f_factor=1

variable_lr=$1

model_shape=$2
model_shapes=('XSv2')

patchifier_embed_dim=$3 # MUST BE =3 for VQGAN (=z_shape), =32 otherwise

shuffle=$4

l1_weight=$5
fft_l1_weight=$6
ssim_weight=$7
l2_weight=$8

exp="i2iwfilm_S1"

max_epochs=600

# seed=0

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "The directory of the script is: $DIR"
i2iwfilm_path="$(dirname "$DIR")/.."
echo "I2IwFiLM path set to: $i2iwfilm_path"

job_ids=()

# seeds to run: 0 to 10
seeds=(0)
# seeds=(0 1 2 3 4 5 6 7 8 9)


for seed in "${seeds[@]}"; do

    for i in "${!src_dtype[@]}"; do

        for model_shape in "${model_shapes[@]}"; do

            if [ "$model_shape" = 'S' ] ; then
                blocks_per_depth='[1,1,1,9]'
                heads_per_depth='[1,2,4,8]'
            elif [ "$model_shape" = 'M' ] ; then
                blocks_per_depth='[2,2,2,4]'
                heads_per_depth='[1,2,4,8]'
            elif [ "$model_shape" = 'L' ] ; then
                blocks_per_depth='[4,6,6,8]'
                heads_per_depth='[1,2,4,8]'
            elif [ "$model_shape" = 'XS' ] ; then
                blocks_per_depth='[2,3,4]'
                heads_per_depth='[2,4,8]'
            elif [ "$model_shape" = 'XSv2' ] ; then
                blocks_per_depth='[2,2,2]'
                heads_per_depth='[16,16,16]'
            elif [ "$model_shape" = 'Mv2' ] ; then
                blocks_per_depth='[2,2,2,2]'
                heads_per_depth='[16,16,16,16]'
            else
                echo "model shape not recognized"
                exit 1
            fi

            src=${src_dtype[$i]}
            tgt=${tgt_dtype[$i]}

            job_name="I2IwFiLM_S1_seed_${seed}"
            
            job_name="f${f_factor}_${job_name}"
            
                                                                # $1   $2   $3   $4    $5          $6        $7         $8            $9              ${10}             ${11}               ${12}             ${13}     ${14}        ${15}         ${16}      ${17}
            echo "sbatch ./scripts/wl2cal/i2iwfilm_S1_train.sh $seed $src $tgt $exp $batch_size $job_name $max_epochs $i2iwfilm_path $variable_lr $blocks_per_depth $heads_per_depth $patchifier_embed_dim $shuffle $l1_weight $fft_l1_weight $ssim_weight $l2_weight"
            job_id=$(sbatch ./scripts/wl2cal/i2iwfilm_S1_train.sh $seed $src $tgt $exp $batch_size $job_name $max_epochs $i2iwfilm_path $variable_lr $blocks_per_depth $heads_per_depth $patchifier_embed_dim $shuffle $l1_weight $fft_l1_weight $ssim_weight $l2_weight| grep -o -E '[0-9]+')

            job_id=$(echo $job_id | grep -o -E '[0-9]+')
            job_ids+=($job_id)

        done

    done

done

echo "job ids: ${job_ids[@]}"