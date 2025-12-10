# BLD-ML

# Pipeline Script Overview
1. SLURM Batch Script (run_job_efficientnet.py or similar)
Purpose: Defines the job submission to the HPC cluster (SCINet/Atlas).

Key functions:
Requests resources (GPU type, CPUs, memory, walltime).
Sets job metadata (name, output logs, account, email notifications).
Activates the Python environment.
Launches the training run with srun python run_training.py ....

2. run_training.py
Purpose: Entry point for model training and evaluation.
Key functions:
Parses command-line arguments (dataset path, epochs, batch size, model type, etc.).
Defines image preprocessing and augmentation pipelines.
Loads datasets (train, val, test) and wraps them with FlipLabels for consistent binary encoding.
Builds the chosen backbone model (EfficientNet, ResNet, MobileNet, Inception).
Calls train_1_binary (from training.py) to run the training loop.
After training, reloads the best model and evaluates on the test set.
Saves metrics (loss, accuracy, precision, recall, F1, AUROC) to CSV files.

3. training.py
Purpose: Implements the training loop for binary classification.
Key functions:
Freezes backbone layers initially (transfer learning).
Sets up optimizer, loss function, and learning rate scheduler.
Supports mixed precision training for efficiency.
Runs epoch-by-epoch training and validation:
Tracks loss, accuracy, precision, recall, F1, AUROC.
Logs metrics to CSV.
Implements early stopping and checkpoint saving.
Optionally unfreezes backbone blocks after a set epoch for fine-tuning.
Saves both best and final model weights.

4. optim_utils.py
Purpose: Utility functions for optimizers and schedulers.
Key functions:
Builds optimizers with differential learning rates (smaller LR for backbone, larger for head).
Provides a ReduceLROnPlateau scheduler to adjust LR when validation metrics plateau.
Re-exports backbone control functions from model_utils.py for convenience.

5. model_utils.py
Purpose: Low-level utilities for model parameter management.
Key functions:
Identifies the classifier head (e.g., classifier, fc) across different architectures.
Freezes backbone parameters while keeping head trainable.
Sets BatchNorm layers to eval mode to stabilize transfer learning.
Unfreezes the last N backbone blocks for progressive fine-tuning.


