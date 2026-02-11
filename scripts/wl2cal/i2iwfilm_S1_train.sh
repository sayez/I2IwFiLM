#!/bin/bash

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=48:00:00

############################################


# print( the current hostname)
echo "hostname: $(hostname)"

# random seed
seed=$1

# source and target domain types: can be 'drawing', 'whitelight', 'calcium'
src=$2
tgt=$3

# experiment name/ config file: can be 'test_vqgan'
exp_file=$4

# batch size
batch_size=$5

# job name
job_name=$6

max_epochs=$7

i2iwfilm_path=$8

variable_lr=$9

blocks_per_depth=${10}
heads_per_depth=${11}

patchifier_embed_dim=${12}

shuffle=${13}

l1_weight=${14}
fft_l1_weight=${15}
ssim_weight=${16}
l2_weight=${17}

output_dir_base="wl2cal/S1"

dt=$(date '+%Y-%m-%d/%H-%M-%S');
# DO NOT USE DT IN THE NAME, IF JOB IS PREEMPTED + RE LAUNCHED,
# IT WILL CREATE A NEW FOLDER EVERY TIME, INSTED OF CONTINUING THE ORIGINAL ONE
hydra_out_dir="./outputs/${output_dir_base}/${job_name}"

echo "seed: $seed, source: $src, target: $tgt, exp: $exp_file, batch_size: $batch_size, job_name: $job_name, max_epochs: $max_epochs"
echo "variable_lr: $variable_lr, blocks_per_depth: $blocks_per_depth , heads_per_depth: $heads_per_depth, patchifier_embed_dim: $patchifier_embed_dim"
echo "shuffle: $shuffle"
echo "l1_weight: $l1_weight, fft_l1_weight=$fft_l1_weight, ssim_weight=$ssim_weight l2_weight=$l2_weight"

export I2IWFILM_PATH=$i2iwfilm_path
echo "I2IwFiLM path: $I2IWFILM_PATH"

echo "OUTPUT DIR: $hydra_out_dir"

export HYDRA_FULL_ERROR=1


if [ "$variable_lr" = true ] ; then
    echo "Using variable learning rate"
    python -m i2iwfilm.train gpus=1 \
                    hydra.run.dir=$hydra_out_dir \
                    trainer.max_epochs=$max_epochs \
                    seed=$seed \
                    exp=$exp_file \
                    src_dtype=$src \
                    tgt_dtype=$tgt \
                    dataset.batch_size=$batch_size \
                    module.loss_weights.0=$l1_weight \
                    module.loss_weights.1=$fft_l1_weight \
                    module.loss_weights.2=$ssim_weight \
                    module.loss_weights.3=$l2_weight \
                    model.num_blocks=$blocks_per_depth \
                    model.heads=$heads_per_depth \
                    model.patchifier_embed_dim=$patchifier_embed_dim \
                    dataset.strategies.kfold.shuffle=$shuffle \
                    ++logger.0.name=$job_name
else
    echo "Using constant learning rate"

    python -m i2iwfilm.train gpus=1 \
                    hydra.run.dir=$hydra_out_dir \
                    trainer.max_epochs=$max_epochs \
                    seed=$seed \
                    exp=$exp_file \
                    src_dtype=$src \
                    tgt_dtype=$tgt \
                    dataset.batch_size=$batch_size\
                    module.loss_weights.0=$l1_weight \
                    module.loss_weights.1=$fft_l1_weight \
                    module.loss_weights.2=$ssim_weight \
                    module.loss_weights.3=$l2_weight \
                    ~module.scheduler \
                    ~module.scheduler_interval \
                    ~scheduler \
                    model.num_blocks=$blocks_per_depth \
                    model.heads=$heads_per_depth \
                    model.patchifier_embed_dim=$patchifier_embed_dim \
                    dataset.strategies.kfold.shuffle=$shuffle \
                    ++logger.0.name=$job_name
fi

            

