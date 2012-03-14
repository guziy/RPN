#!/bin/bash

## Usage: qsub sge_submit.sh
##$ -j y
#$ -cwd
#$ -e /home/huziy/skynet3_rech1/error
#$ -o /home/huziy/skynet3_rech1/log
#$ -S /bin/sh
#$ -M guziy.sasha@gmail.com
#$ -m b
#$ -q q_skynet3
## 6 shared memory processes????
#$ -pe shm 10
##Use my current environment variables
#$ -V
which python
export PYTHONPATH=$PYTHONPATH:./src
export OMP_NUM_THREADS=$NSLOTS
python src/permafrost/save_field_from_rpn_to_netcdf.py 
