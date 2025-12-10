#!/bin/bash
#SBATCH --job-name=nematode_train
#SBATCH --output=slurm-%j.out   # STDOUT/STDERR file (%j = job ID)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=8       # adjust to your dataloader workers
#SBATCH --mem=32G
#SBATCH --time=24:00:00         # max run time (HH:MM:SS)
#SBATCH --mail-type=END,FAIL    # optional: email on end/fail
#SBATCH --mail-user=Benjamin.Waldo@usda.gov

# Load modules / activate env
module load cuda/11.8            # or whatever your cluster uses
source /project/nematode_ml/envs/ipykernel

# Run the training
python train_nematode_binary_two_phase.py
