#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --nodes=1 --mem=102400              #
#SBATCH --cpus-per-task=32 --ntasks-per-node=1               #
#SBATCH --time=02-00:00           # time (DD-HH:MM)

. s.ssmuse.dot pylibrmn_deps
. /project/6004670/huziy/Python/software/2017/Core/miniconda3/4.3.27/bin/activate root


cd $SLURM_SUBMIT_DIR
export PYTHONPATH=./src:$PYTHONPATH

#can be either current or future


opt=${1:-current}

time python -u src/lake_effect_snow/hles_cc/calculate_hles_current_future_HL.py ${opt} 25  >& hles_hl_gl_${opt}.log 


