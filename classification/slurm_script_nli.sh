#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:2
#SBATCH --time=01:00:00
#SBATCH --job-name=classify_nli
#SBATCH --export=NONE
#SBATCH --output=out/out_nli.txt

unset SLURM_EXPORT_ENV

module load python

source my_venv/bin/activate

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80


python run_nli.py roberta-large-mnli test_data/test_data_mistakes.json roberta_finetuned_imppres_pre_optimum

## fine-tuning

## IMPPRES
# python run_nli.py facebook/bart-large-mnli test_data/test_data_mistakes.json bart_finetuned_imppres_pre_optimum -ft imppres
# python run_nli.py roberta-large-mnli test_data/test_data_mistakes.json roberta_finetuned_imppres_pre_optimum -ft imppres
# python run_nli.py MoritzLaurer/deberta-v3-large-zeroshot-v2.0 test_data/test_data_mistakes.json deberta_finetuned_imppres_pre_optimum -ft imppres -b

# ## INLI
# python run_nli.py facebook/bart-large-mnli test_data/test_data_mistakes.json bart_finetuned_inli -ft inli/INLI\ Data/train.csv,inli/INLI\ Data/val.csv
# python run_nli.py roberta-large-mnli test_data/test_data_mistakes.json roberta_finetuned_inli -ft inli/INLI\ Data/train.csv,inli/INLI\ Data/val.csv
# python run_nli.py MoritzLaurer/deberta-v3-large-zeroshot-v2.0 test_data/test_data_mistakes.json deberta_finetuned_inli -ft inli/INLI\ Data/train.csv,inli/INLI\ Data/val.csv -b

## 200 HELD OUT

# python run_nli.py nli-ft/bart_finetuned_imppres_pre_optimum test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_bart_optimum_c.json -c
# python run_nli.py nli-ft/roberta_finetuned_imppres_pre_quest test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_quest_roberta_c.json -c
# python run_nli.py nli-ft/deberta_finetuned_imppres_pre_optimum test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_deberta_optimum_c.json -c

# python run_nli.py nli-ft/bart_finetuned_imppres_pre test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_bart_c.json -c
# python run_nli.py nli-ft/roberta_finetuned_imppres_pre test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_roberta_c.json -c
# python run_nli.py nli-ft/deberta_finetuned_imppres_pre test_data/test_data_wo_few_shots_noties_merged.json predictions_mergedstudy_finetuned_imppres_pre_deberta_c.json -c

## 108 CROSS VALIDATION?