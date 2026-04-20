#!/bin/bash

# models=("gpt" "deepseek" "qwen" "mistral" "mixtral" "llama")
models=("llama" "qwen" "mistral")
nli_models=("bart" "roberta" "deberta")
# nli_models=("bart")

for model in "${models[@]}"; do
    python3 evaluate_predictions.py implicit_data/gold_standard.json "test/${model}" evaluation_classification_${model}.json
    # python3 evaluate_predictions.py implicit_data/gold_standard.json "test/${model}" evaluation_classification_${model}.json -m "mistakes_${model}.json" -t
    python3 evaluate_predictions.py implicit_data/gold_standard.json "out_of_domain_test/${model}" evaluation_classification_${model}.json
    # python3 evaluate_predictions.py implicit_data/gold_standard.json "out_of_domain_test/${model}" evaluation_classification_${model}.json -m "out_of_domain_test/mistakes/mistakes_${model}.json" -t
done

# for nli_model in "${nli_models[@]}"; do
#     python3 evaluate_predictions.py implicit_data/gold_standard.json "test/${nli_model}" evaluation_classification_${nli_model}.json -nli -sp 0 -p
# #     python3 evaluate_predictions.py implicit_data/gold_standard.json "test/${nli_model}" evaluation_classification_${nli_model}.json -m "test/mistakes/mistakes_${nli_model}.json" -nli -sp 0 -t
#     python3 evaluate_predictions.py implicit_data/gold_standard.json "out_of_domain_test/${nli_model}" evaluation_classification_${nli_model}.json -nli -sp 0 -p
# #     python3 evaluate_predictions.py implicit_data/gold_standard.json "out_of_domain_test/${nli_model}" evaluation_classification_${nli_model}.json -m "out_of_domain_test/mistakes/mistakes_${nli_model}.json" -nli -sp 0 -t
# done
