#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=48:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -l gpu=1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -pe sharedmem 1
#$ -l h_vmem=16G

# Say hello
echo "Hellote"




# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${hostname}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Directory setup
# ===================

SCRATCH_DISK=/exports/eddie/scratch/
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
echo "Let's make ${SCRATCH_HOME}"
mkdir -p ${SCRATCH_HOME}
echo "and let's check it"
echo `ls ${SCRATCH_HOME}`
echo "Let's make ${SCRATCH_HOME}/philoso-py/output"
mkdir -p ${SCRATCH_HOME}/philoso-py/output
echo "and let's check it"
echo `ls ${SCRATCH_HOME}/philoso-py/`

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"
# Initialise the environment modules and load CUDA version 12.1.1
source /etc/profile.d/modules.sh

echo "loading modules"
module load cuda/12.1.1
# module load python/3.12.9
module load anaconda/2024.02


# ===================
# Conda environment setup
# ===================

conda config --add envs_dirs ${SCRATCH_HOME}/philoso-py/anaconda/envs
conda config --add pkgs_dirs ${SCRATCH_HOME}/philoso-py/anaconda/pkgs

# Create python virtual environment and install modules:
ENV_NAME=philos_env
echo "Create and activate ${ENV_NAME} with reqs"
conda create --name ${ENV_NAME} python=3.11 # --file requirements.txt
echo "there, I created it"
conda activate philos_env
echo "and activated it"
echo `python -V`
pip install -r requirements.txt
python qtest.py

# Make available all commands on $PATH as on headnode
# source ~/.bashrc # ???
# echo "Make available all commands on $PATH as on headnode"
# Make script bail out after first error
# set -e # ???

# Make your own folder on the node's scratch disk



# Create and activate your conda environment

# # create venv, ~20 minutes
# echo "make environment with python 3.10"
# /opt/conda/bin/python3.10 -m venv "${ENV_NAME}"
# echo "activate env"
#################### source "${ENV_NAME}/bin/activate"
# which python
# echo "upgrading pip"
# # ~20 minutes
# pip install --upgrade pip
# echo "install packages"
# pip install -r requirements.txt



# =================================
# Move input data to scratch disk
# =================================

################### echo "Not moving data to $SCRATCH_HOME because philoso-py doesn't use training data"

# data directory path on the DFS
################### src_path=/home/s0454279/philoso-py

# # input data directory path on the scratch disk of the node
###################dest_path=${SCRATCH_HOME}/philoso-py
# mkdir -p ${dest_path}  # make it if required
# # Important notes about rsync:
# # * the --compress option is going to compress the data before transfer to send
# #   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# # * the final slash at the end of ${src_path}/ is important if you want to send
# #   its contents, rather than the directory itself. For example, without a
# #   final slash here, we would create an extra directory at the destination:
# #       ${SCRATCH_HOME}/project_name/data/input/input
# # * for more about the (endless) rsync options, see the docs:
# #       https://download.samba.org/pub/rsync/rsync.html

# =============================================
# No data import needed for these experiments
# ============================================= 

# rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

#


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
