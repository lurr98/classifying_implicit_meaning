#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --job-name=classify_gpt
#SBATCH --export=NONE
#SBATCH --output=out/out_gpt.txt

unset SLURM_EXPORT_ENV

module load python

source ../my_venv/bin/activate

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python3 theoretical_frameworks.py ../samples/samples_merged.json rst rst_parsed_test.json
# python3 rst.py ../samples/samples_merged.json rst rst_parsed_test.json
# python3 rst.py ../samples/samples_merged.json framenet framenet_parsed.json