#!/bin/bash
#SBATCH --mem=30G
#SBATCH --output=/home/pbelcak/slt/log/%j.out
#SBATCH --error=/home/pbelcak/slt/log/%j.err
#SBATCH --mail-type=NONE                            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
	echo 'Failed to create temp directory' >&2
	exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

USER=pbelcak

# Binary or script to execute
PYTHONPATH=/home/${USER} python -m slt --job-id=${SLURM_JOB_ID} --input-path=/itet-stor/${USER}/net_scratch/slt/slt_data --output-path=/itet-stor/${USER}/net_scratch/slt/models $@

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0