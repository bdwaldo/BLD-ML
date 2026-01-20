#!/bin/bash -l
# Use bash as the shell, with the -l option to start as a login shell
# (ensures environment variables and modules are properly initialized)

## MobileNetV3-Large
# A comment describing the purpose of the job (training MobileNetV3-Large)

#SBATCH --partition=gpu-v100
# Submit the job to the 'gpu-v100' partition (queue) which has NVIDIA V100 GPUs

#SBATCH --gres=gpu:v100:1
# Request 1 V100 GPU as a generic resource

#SBATCH --cpus-per-task=8
# Allocate 8 CPU cores for this task (for data loading, preprocessing, etc.)

#SBATCH --qos=normal
# Use the 'normal' quality of service (QoS) setting, which may affect priority

#SBATCH --job-name=BLD_train_mobilenet
# Name of the job, useful for tracking in the queue

#SBATCH --output=BLD_train_mobilenet-%j.out
# Save standard output to a file named with the job ID (%j expands to job number)

#SBATCH --account=nematode_ml
# Charge compute time to the 'nematode_ml' project account

#SBATCH --mail-user=Benjamin.Waldo@usda.gov
# Email address to send job notifications

#SBATCH --mail-type=END,FAIL
# Send email when the job ends or fails

#SBATCH --time=08:00:00
# Maximum wall-clock time: 8 hours

#SBATCH --nodes=1
# Request 1 compute node

#SBATCH --ntasks=1
# Run 1 task (single process). Combined with --cpus-per-task, this means
# 1 process using 8 CPU cores and 1 GPU.

module purge
# Clear all loaded modules to start with a clean environment

# Load modules / activate env
source /project/nematode_ml/envs/ipykernel/bin/activate
# Activate a Python virtual environment located in the project directory

# MobileNetV3-Large training command
srun /project/nematode_ml/envs/ipykernel/bin/python \
    /90daydata/nematode_ml/BLD/nematode_project/run_training.py \
    --data-root /90daydata/nematode_ml/BLD/NematodeDataset \
    --num-workers 8 \
    --epochs 100 \
    --mixed-precision \
    --model mobilenet_v3_large
# Use 'srun' to launch the Python training script (run_training.py) with SLURM.
# Arguments:
#   --data-root: path to dataset
#   --num-workers: number of data loader workers (matches CPUs requested)
#   --epochs: train for 100 epochs
#   --mixed-precision: enable mixed precision training (faster, less memory)
#   --model: specify MobileNetV3-Large architecture
