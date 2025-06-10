#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 6 days:
#$ -l h_rt=144:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request 72 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=144G
# which json files to use in array job
#$ -t 3-3

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

SCRATCH_DISK=/exports/eddie/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
SCRATCH_PHILOSOPY=${SCRATCH_HOME}/philoso-py
SCRATCH_OUTPUTS=${SCRATCH_PHILOSOPY}/output
echo "Let's make ${SCRATCH_OUTPUTS}"
mkdir -p ${SCRATCH_OUTPUTS}
echo "and let's check it"
echo `ls ${SCRATCH_PHILOSOPY}`



# Target directory
PHILOSOPY_DIR=$HOME/philoso-py
JSON_DIR=${PHILOSOPY_DIR}/model_json
OUTPUTS_TARG_DIR=${PHILOSOPY_DIR}/output

# Get list of files in target directory
files=$(ls -1 ${JSON_DIR}/*)


JSON_FILE=$(echo "${files}" | sed -n ${SGE_TASK_ID}p)
echo "${JSON_FILE} is the json"
model_id=`egrep -o "[\"']model_id[\"']: [\"']([0-9a-zA-Z\-_]*)[\"']" ${JSON_FILE} | egrep -o ": [\"']([0-9a-zA-Z\-_]*)[\"']" | egrep -o "[0-9a-zA-Z\-_]*"`
echo "Model: ${model_id}"

SCRATCH_MODEL_OUTPUT=${SCRATCH_OUTPUTS}/${model_id}
OUT_TARG_DIR=${OUTPUTS_TARG_DIR}/${model_id}
mkdir -p ${OUT_TARG_DIR}


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"
# Initialise the environment modules and load CUDA version 12.1.1
source /etc/profile.d/modules.sh

echo "loading modules"
# module load python/3.12.9
module load anaconda/2024.02


# ===================
# Conda environment setup
# ===================

conda config --add envs_dirs ${SCRATCH_HOME}/philoso-py/anaconda/envs
conda config --add pkgs_dirs ${SCRATCH_HOME}/philoso-py/anaconda/pkgs

# Create python virtual environment if needed:
ENV_NAME=philos_env
ENV_LIST=$(conda env list)

if [[ "${ENV_LIST}" != *"${ENV_NAME}"* ]]; then
    echo "Create and activate ${ENV_NAME} with reqs"
    conda create --name ${ENV_NAME} python=3.11 # --file requirements.txt
    echo "there, I created it"
else
    echo "${ENV_NAME} already exists"
fi

# Activate env and install modules
conda activate philos_env
conda install pip
echo "and I activated it"
echo "Today's flavour of Python is:"
echo `python -V`
pip install -r requirements.txt --no-cache-dir





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


# ==============================
# Finally, run the experiment!
# ==============================
echo 'OK, here we go'

python philoso_py.py ${JSON_FILE} -o ${SCRATCH_PHILOSOPY}

# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

#
SCRATCH_MODEL_OUTPUT_TGZ="${SCRATCH_MODEL_OUTPUT}.tar.gz"
tar -czvf ${SCRATCH_MODEL_OUTPUT_TGZ} ${SCRATCH_MODEL_OUTPUT}

rsync --archive --update --compress --progress ${SCRATCH_MODEL_OUTPUT_TGZ} ${OUT_TARG_DIR}

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
