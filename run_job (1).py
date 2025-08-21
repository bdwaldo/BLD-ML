#!/bin/bash -l

#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=normal
#SBATCH --job-name=BLD_train
#SBATCH --output=BLD_train-%j.out
#SBATCH --account=nematode_ml
#SBATCH --mail-user=Benjamin.Waldo@usda.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge

# Load modules / activate env
source /project/nematode_ml/envs/ipykernel/bin/activate

# Run the training
srun /project/nematode_ml/envs/ipykernel/bin/python /90daydata/nematode_ml/BLD/nematode_project/run_training.py \
    --data-root /90daydata/nematode_ml/BLD/NematodeDataset \
    --num-workers 8 \
    --epochs 25 \
    --mixed-precision

