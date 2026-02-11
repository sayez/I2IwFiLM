# Mitigating hallucination with non-adversarial strategies for image-to-image translation in solar physics


## Installation

1) Clone this repository to desired location.

2) Create a virtual environment using python 3.10 .

    ```
    conda create -n 'i2iwfilm' python=3.10
    conda activate i2iwfilm
    ```

3) From root of the cloned repository, install i2iwfilm as a package.
    
    ```
    pip install -e .
    ```

    The editable (-e) option allows creation and use of functions after package installation. If you create a new function, you must restart open jupyter notebooks to reload new package function tree.

4) IMPORTANT: Adapt the paths to local filesystem for datasets/outputs (using symbolic links if needed) - I2IwFiLM expects the following structure:

    ```
    I2IwFiLM
    | -datasets
    | | - WL2CAL *
    | | | - test ...
    | | | - train ...
    | | - AIA2HMI *
    | | | - test
    | | | - train
    | - outputs **
    ...

    ```

    `*` = Symbolic link to directories with same name in storage.
    `**` = Symbolic link to directory where trained models can be stored.

## Use Inference notebooks

### A) Directly on computing machine

1) In the root directory of the cloned repository, launch a jupyter server

    ```
    jupyter-notebook
    ```

2) Access the jupyter interface by clicking the link shown in terminal.

3) In the interface, open the desired notebook.


### B) On a client machine

1) (REMOTE SERVER) Ensure that the computing machine launched a jupyter server using command. Consider using tmux to keep the server alive when disconnected. (Change XXXX to a free port number)

    ```
    jupyter-notebook --no-browser --port=XXXX
    ```
    
    **-> Notice the url link starting with localhost. <-**


2) (CLIENT) To access the jupyter interface from client machine, setup port forwarding between local port and server port.

    ```
    ssh -N -L XXXX:localhost:XXXX user_name@server_name
    ```

3) (CLIENT) Follow the url link shown at **step 1** on a web browser on client machine. This opens jupyter interface.


3) In the interface, open the desired notebook.



## Train networks

This repository relying on the Hydra python module, trainings typically follow experiments described in  configuration files in i2iwfilm/conf/exp/XXXX.yaml

The typical bash command (**!launch from the root folder of the repo!**) to run experiment EXP.yaml is:

```
python -m i2iwfilm.train exp=EXP hydra.run.dir=OUT_DIR seed=0 gpus=1
```

Hydra offers the possibility to override the configuration elements by adding/redifining them in the launch command:

```
 .... dataset.batch_size=32  ++dataset.num_workers=16
```
(the ++ sign means 'add configuration item or override if already exists')


Therefore, to run new experiments either create new configuration files to fit your need, or use the exitsting ones and modify the bash command accordingly.

In the scripts/ folder, you'll find bash files able to launch trainings divided in two kinds:
 1) XXX.sh files: scripts defining output folder and starting training by calling the training command with some configuration overrides given as ordered parameters.Those files are compatible to be run as a slurm job (SBATCH comments at the top of the file).
 2) XXX_launcher.sh files: bash scripts defining possibly several sets of training overrides and displaying calls to XXX.sh scripts to be launched as slurm jobs, once per set (with the corresponding parameters values).

Better than an explanation, here are examples to launch both training phases.


### I2IwFiLM Network training

The training of I2IwFiLM networks is split in two phases: the first uses both source and target modalities as an input and the second learns to rely on source modality only.

For each phase, XXX.sh and XXX_launcher.sh files can be used.

#### Phase 1

To train the first stage, launch the following command in a terminal in the I2IwFiLM repository:

```
bash ./scripts/wl2cal/i2iwfilm_S1_train.sh SEED SOURCE_MODALITY TARGET_MODALITY EXPERIMENT_FILE BATCH_SIZE RUN_NAME NUM_EPOCH PATH_TO_I2IWFILM_REPO USE_VARIABLE_LEARNING_RATE_BOOLEAN BLOCKS_PER_DEPTH HEADS_PER_DEPTH PATCHIFIER_DIMS SHUFFLE_DATA_BOOLEAN L1_WEIGHT FFT_L1_WEIGHT SSIM_WEIGHT L2_WEIGHT
```

Here is a breakdown of the arguments for the script:

- SEED: A random seed for reproducibility.
- SOURCE_MODALITY: The source modality for the image-to-image translation (e.g., "whitelight").
- TARGET_MODALITY: The target modality for the image-to-image translation (e.g., "calcium").
- EXPERIMENT_FILE: The name of the experiment configuration file (e.g., "i2iwfilm_S1.yaml").
- BATCH_SIZE: The batch size to use during training.
- RUN_NAME: A name for the training run, used for logging and saving checkpoints.
- NUM_EPOCH: The number of epochs to train the model.
- PATH_TO_I2IWFILM_REPO: The file path to the I2IWFILM repository on your local machine.
- USE_VARIABLE_LEARNING_RATE_BOOLEAN: A boolean value indicating whether to use a variable learning rate during training.
- BLOCKS_PER_DEPTH: The number of transformer blocks per depth in the model architecture.
- HEADS_PER_DEPTH: The number of attention heads per depth in the model architecture.
- PATCHIFIER_DIMS: The dimensions of the patchifier output used in the model.
- SHUFFLE_DATA_BOOLEAN: A boolean value indicating whether to shuffle the training data during training.
- L1_WEIGHT: The weight for the L1 loss component in the training objective.
- FFT_L1_WEIGHT: The weight for the FFT L1 loss component in the training objective.
- SSIM_WEIGHT: The weight for the SSIM loss component in the training objective.
- L2_WEIGHT: The weight for the L2 loss component in the training objective.


#### Phase 2

To train the second stage, launch the following command in a terminal in the I2IwFiLM repository:

```
bash ./scripts/wl2cal/i2iwfilm_S2_train.sh RUN_NAME SOURCE_MODALITY TARGET_MODALITY PATH_TO_I2IWFILM_REPO EXPERIMENT_FILE NUM_EPOCH BATCH_SIZE SHUFFLE_DATA_BOOLEAN USE_VARIABLE_LEARNING_RATE_BOOLEAN PATH_TO_S1_RUN_DIR REL_PATH_TO_CKPT_IN_S1 L1_WEIGHT FFT_L1_WEIGHT SSIM_WEIGHT L2_WEIGHT GVP_L1_WEIGHT MLP_HID_LAY_SIZES SEED
```

Here is a breakdown of the parameters:

- RUN_NAME: A name for the training run, used for logging and saving checkpoints.
- SOURCE_MODALITY: The source modality for the image-to-image translation (e.g., "whitelight").
- TARGET_MODALITY: The target modality for the image-to-image translation (e.g., "calcium").
- PATH_TO_I2IWFILM_REPO: The file path to the I2IwFiLM repository on your local machine.
- EXPERIMENT_FILE: The name of the experiment configuration file (e.g., "i2iwfilm_S2.yaml").
- NUM_EPOCH: The number of epochs to train the model.
- BATCH_SIZE: The batch size to use during training.
- SHUFFLE_DATA_BOOLEAN: A boolean value indicating whether to shuffle the training data.
- USE_VARIABLE_LEARNING_RATE_BOOLEAN: A boolean value indicating whether to use a variable learning rate during training.
- PATH_TO_S1_RUN_DIR: The file path to the directory containing the Stage 1 run, which may be used for initializing the model or for other purposes during Stage 2 training.
- REL_PATH_TO_CKPT_IN_S1: The relative file path to the checkpoint within the Stage 1 run directory that should be used for initializing the model for Stage 2 training.
- L1_WEIGHT: The weight for the L1 loss component in the training objective.
- FFT_L1_WEIGHT: The weight for the FFT L1 loss component in the training objective.
- SSIM_WEIGHT: The weight for the SSIM loss component in the training objective.
- L2_WEIGHT: The weight for the L2 loss component in the training objective.
- GVP_L1_WEIGHT: The weight for the GVP L1 loss component in the training objective.
- MLP_HID_LAY_SIZES: The sizes of the hidden layers in the MLP (Multi-Layer Perceptron) used in the model architecture, specified as a comma-separated list (e.g., "[128,128,128]").
- SEED: The random seed to use for reproducibility during training.



