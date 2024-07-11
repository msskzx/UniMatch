#!/bin/bash
#SBATCH --job-name=exp4
#SBATCH --output=outputs/out/ukbb/supervised/unet/seg_only_mid_slices/exp4-%A.out  # Standard output  %A adds the job id
#SBATCH --error=outputs/err/ukbb/test/supervised/unet/seg_only_mid_slices/exp4-%A.err  # Standard error 
#SBATCH --time=0-01:00:00  # Limit time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=16  # Number of CPUs (limit 24 per GPU)
#SBATCH --ntasks 1
#SBATCH --mem=20G  # Memory in GB (limit 48GB per GPU)

# load python module
module load python/anaconda3
source activate
conda deactivate # If env is loaded, conda won't activate the environment.
conda activate unimatch # If this does not work, try 'source activate ptl'

now=$(date +"%Y%m%d_%H%M%S")
exp='4'
seed='83'
control='ethn'
method='supervised'
dataset='ukbb'
job=exp${exp}_$control

python -u inference.py --control=$control --seed=$seed --exp=$exp --method=$method --dataset=$dataset