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

# job name
job_name=$1

# source and target domain types: can be 'drawing', 'whitelight', 'calcium'
src=$2
tgt=$3

i2iwfilm_path=$4
exp_file=$5

max_epochs=$6
batch_size=$7
shuffle=$8

variable_lr=$9

s1_run_dir=${10}
s1_model_path=${11}

l1_weight=${12}
fft_l1_weight=${13}
ssim_weight=${14}
l2_weight=${15}

GVP_l1_weight=${16}

mlp_hidden=${17}

# random seed
seed=${18}

output_dir_base="wl2cal/S2"

dt=$(date '+%Y-%m-%d/%H-%M-%S');
# DO NOT USE DT IN THE NAME, IF JOB IS PREEMPTED + RE LAUNCHED,
# IT WILL CREATE A NEW FOLDER EVERY TIME, INSTED OF CONTINUING THE ORIGINAL ONE
hydra_out_dir="./outputs/${output_dir_base}/${job_name}"

echo "seed: $seed, source: $src, target: $tgt, exp: $exp_file, batch_size: $batch_size, job_name: $job_name, max_epochs: $max_epochs"
echo "variable_lr: $variable_lr"
echo "shuffle: $shuffle"
echo "l1_weight: $l1_weight, fft_l1_weight=$fft_l1_weight, ssim_weight=$ssim_weight, l2_weight=$l2_weight GVP_l1_weight=$GVP_l1_weight"
echo "run_dir: $s1_run_dir , model_path: $s1_model_path"
echo "mlp_hidden: $mlp_hidden"

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
                    module.diff_former_loss_weights.0=$l1_weight \
                    module.diff_former_loss_weights.1=$fft_l1_weight \
                    module.diff_former_loss_weights.2=$ssim_weight \
                    module.diff_former_loss_weights.3=$l2_weight \
                    module.GVP_loss_weights.0=$GVP_l1_weight \
                    model.s1_run_dir=$s1_run_dir \
                    model.s1_model_path=$s1_model_path \
                    model.model_s2.mlp_hidden=$mlp_hidden \
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
                    module.diff_former_loss_weights.0=$l1_weight \
                    module.diff_former_loss_weights.1=$fft_l1_weight \
                    module.diff_former_loss_weights.2=$ssim_weight \
                    module.diff_former_loss_weights.3=$l2_weight \
                    module.GVP_loss_weights.0=$GVP_l1_weight \
                    ~module.scheduler \
                    ~module.scheduler_interval \
                    ~scheduler \
                    model.s1_run_dir=$s1_run_dir \
                    model.s1_model_path=$s1_model_path \
                    model.model_s2.mlp_hidden=$mlp_hidden \
                    dataset.strategies.kfold.shuffle=$shuffle \
                    ++logger.0.name=$job_name
fi

            

