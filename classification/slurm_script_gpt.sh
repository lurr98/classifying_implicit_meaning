#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=05:00:00
#SBATCH --job-name=classify_gpt
#SBATCH --export=NONE
#SBATCH --output=out/out_gpt.txt

unset SLURM_EXPORT_ENV

module load python

source my_venv/bin/activate

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

## 200 HELD OUT

# noties
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4.1-mini predictions_gpt4.1mini_zero_noties_2507.json -z
# 
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o-mini predictions_gpt4o-mini_zero_noties_1007.json -z
# 
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o-mini predictions_gpt4o-mini_few2_noties_2507.json -f 2
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o-mini predictions_gpt4o-mini_few4_noties_2507.json -f 4
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o-mini predictions_gpt4o-mini_few8_noties_2507.json -f 8
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o-mini predictions_gpt4o-mini_few16_noties_2507.json -f 16


# noties
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_zero_noties_2507.json -z
# 
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_zero_noties_1007.json -z

# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_few2_noties_2507.json -f 2
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_few4_noties_2507.json -f 4
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_few8_noties_2507.json -f 8
# python classify_implicit_data.py test_data/test_data_wo_few_shots_noties_merged.json api.txt gpt-4o predictions_gpt4o_few16_noties_2507.json -f 16
# ties
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txtgpt-4.1 predictions_gpt4.1_zero_1007.json -z -t
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txtgpt-4.1 predictions_gpt4.1_few2_1007.json -f 2 -t

# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_zero_1007.json -z -t
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_zero_noties_1007.json -z

# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_few2_1007.json -f 2 -t
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_few4_1007.json -f 4 -t
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_few8_1007.json -f 8 -t
# python classify_implicit_data.py test_data/test_data_wo_few_shots_merged.json api.txt gpt-4.1 predictions_gpt4.1_few16_1007.json -f 16 -t
# 


## 108 CROSS VALIDATION

# # 3005 as test
# python classify_implicit_data.py test_data/test_samples_3005.json api.txt gpt-4o-mini predictions_gpt4o-mini_zero_noties_3005_2507.json -z
# python classify_implicit_data.py test_data/test_samples_1806.json api.txt gpt-4o-mini predictions_gpt4o-mini_zero_noties_1806_2507.json -z
# #          
# # ## RUN THIS AGAIN!!!
# # echo "Starting"
# python classify_implicit_data.py test_data/test_samples_3005.json api.txt gpt-4o-mini predictions_gpt4o-mini_few_noties_3005_2507.json -f 108_1806
# # echo "Done with first one"
# python classify_implicit_data.py test_data/test_samples_1806.json api.txt gpt-4o-mini predictions_gpt4o-mini_few_noties_1806_2507.json -f 108_3005