#!/bin/bash
#SBATCH --job-name=ukbb_unimatch_unet_exp4
#SBATCH --output=outputs/out/ukbb/ukbb_unimatch_unet_exp4-%A.out  # Standard output  %A adds the job id
#SBATCH --error=outputs/err/ukbb/test/ukbb_unimatch_unet_exp4-%A.err  # Standard error 
#SBATCH --time=0-01:00:00  # Limit time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=16  # Number of CPUs (limit 24 per GPU)
#SBATCH --ntasks 1
#SBATCH --mem=20G  # Memory in GB (limit 48GB per GPU)
#SBATCH --partition=master
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.khattab@tum.de

# load python module
module load python/anaconda3
source activate
conda deactivate # If env is loaded, conda won't activate the environment.
conda activate unimatch # If this does not work, try 'source activate ptl'

now=$(date +"%Y%m%d_%H%M%S")
exp_num='4'
seed='83'
control='ethn'
job=ukbb_unimatch_unet_exp${exp_num}_$control
config=configs/ukbb/test/exp$exp_num/seed$seed/$control.yaml

python -u inference.py --config=$config