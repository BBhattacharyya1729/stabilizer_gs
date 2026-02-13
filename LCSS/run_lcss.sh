#!/bin/bash

#SBATCH --job-name=LCSS_job
#SBATCH --output=LCSS_output.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=100G


python test.py
