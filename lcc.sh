#! /bin/bash
#SBATCH --time=01:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=my_test_job # Job name
#SBATCH --ntasks=1 # Number of cores for the job. Same as SBATCH -n 1
#SBATCH --partition=P4V12_SKY32M192_L # Partition/queue to runthe job in. (REQUIRED)
#SBATCH -e logs_lcc/slurm-%j.err # Error file for this job.
#SBATCH -o logs_lcc/slurm-%j.out # Output file for this job.
#SBATCH -A gel_msi290_s22cs685 # Project allocation account name
#SBATCH --gres=gpu:1

module load ccs/Miniconda3
source activate ./envs
python train_intent.py