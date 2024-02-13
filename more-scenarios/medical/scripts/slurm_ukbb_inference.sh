#!/bin/bash
#SBATCH --job-name=ukbb_unimatch_unet_7
#SBATCH --output=outputs/out/ukbb/ukbb_unimatch_unet_7-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=outputs/err/ukbb/ukbb_unimatch_unet_7-%A.err  # Standard error of the script
#SBATCH --time=0-01:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=16  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --ntasks 1
#SBATCH --mem=20G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --partition=master
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.khattab@tum.de

# # load python module
module load python/anaconda3
source activate
# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate unimatch # If this does not work, try 'source activate ptl'

now=$(date +"%Y%m%d_%H%M%S")
job='ukbb_unimatch_unet_7'

python -u ukbb_inference.py