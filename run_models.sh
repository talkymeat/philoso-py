#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# Template for running an sbatch arrayjob with a file containing a list of
# commands to run. Copy this, remove the .template, and edit as you wish to
# fit your needs.
# 
# Assuming this file has been edited and renamed slurm_arrayjob.sh, here's an
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```
echo "Hellote"

# ====================
# Options for sbatch
# ====================
# FMI about options, see https://slurm.schedmd.com/sbatch.html
# N.B. options supplied on the command line will overwrite these set here

# *** To set any of these options, remove the first comment hash '# ' ***
# i.e. `# # SBATCH ...` -> `#SBATCH ...`

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
# #SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# #SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
# #SBATCH --cpus-per-task=4

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-04:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
# #SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]


# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


echo `nvidia-smi`
echo `ls /disk/scratch`
# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc
echo "Make available all commands on $PATH as on headnode"
# Make script bail out after first error
set -e
echo "Make script bail out after first error"

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
echo "echo echo echo"
# SCRATCH_DISK=/home # womp womp scratch is full and unwritable
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
echo "Let's make ${SCRATCH_HOME}"
mkdir -p ${SCRATCH_HOME}
echo "and let's check it"
echo `ls /disk/scratch`
echo "Let's make ${SCRATCH_HOME}/philoso-py/output"
mkdir -p ${SCRATCH_HOME}/philoso-py/output
echo "and let's check it"
echo `ls ${SCRATCH_HOME}/philoso-py/`

# Create and activate your conda environment
ENV_NAME=philos_env
# # create venv, ~20 minutes
# echo "make environment with python 3.10"
# /opt/conda/bin/python3.10 -m venv "${ENV_NAME}"
# echo "activate env"
source "${ENV_NAME}/bin/activate"
# which python
# echo "upgrading pip"
# # ~20 minutes
# pip install --upgrade pip
# echo "install packages"
# pip install -r requirements.txt



# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Not moving data to $SCRATCH_HOME because philoso-py doesn't use training data"

# data directory path on the DFS
src_path=/home/s0454279/philoso-py

# # input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/philoso-py
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

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

out_src_path=${SCRATCH_HOME}/philoso-py/output
out_dest_path=${src_path}/output
echo `ls ${out_src_path}`
echo `ls ${src_path}`
rsync --archive --update --compress --progress ${out_src_path}/ ${out_dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
