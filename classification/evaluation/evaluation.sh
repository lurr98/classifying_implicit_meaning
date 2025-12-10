## Baselines

# on 200 HELD OUT
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/baselines/predictions_majority_baseline.json evaluation_classification_final.json -nt -p
# echo "first one done"
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/baselines/predictions_adjective_baseline.json evaluation_classification_final.json -nt -p

## Linear Models

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_svm_pos.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_svm_lex.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_svm_both.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_dt_pos.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_dt_lex.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_dt_both.json evaluation_classification_linear.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/leave_one_out/linear/predictions_dt_108_pos.json evaluation_classification_linear.json -nt -p

## GPT4o

# # on 200 HELD OUT
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_zero_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few2_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few4_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few8_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few16_1007.json evaluation_classification.json -nt -cc -p
# 
# # noties
python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/depr/predictions_gpt4o_zero_noties_2507.json evaluation_classification_final.json -nt -m mistakes/mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o-mini_few2_noties_2507.json evaluation_classification_final.json -nt -m mistakes.json
python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/depr/predictions_gpt4o_few4_noties_2507.json evaluation_classification_final.json -nt -m mistakes/mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o-mini_few8_noties_2507.json evaluation_classification_final.json -nt -m mistakes.json
python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/depr/predictions_gpt4o_few16_noties_2507.json evaluation_classification_final.json -nt -m mistakes/mistakes.json

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_zero_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few2_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few4_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few8_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few16_noties_1007.json evaluation_classification.json -nt -cc -p
# 
# # on kept samples
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few16_1007.json evaluation_classification.json -p
# 
# # on discarded samples
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4o_few16_1007.json evaluation_classification.json -p

# # on 1806 CROSS VALIDATION
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o_zero_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o_zero_3005.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o_few_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o_few_3005.json evaluation_classification_108.json -nt -cc -p

# # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o-mini_zero_noties_1806_2507.json,predictions/classification/108_cross_validation/GPT/predictions_gpt4o-mini_zero_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4o-mini_few_noties_1806_2507.json,predictions/classification/108_cross_validation/GPT/predictions_gpt4o-mini_few_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv

# ## GPT4.1
# 
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_zero_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few2_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few4_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few8_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few16_1007.json evaluation_classification.json -nt -cc -p
# 
# # # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1mini_zero_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1mini_few2_noties_2507.json evaluation_classification_final.json -nt -p -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1mini_few4_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1mini_few8_noties_2507.json evaluation_classification_final.json -nt -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1mini_few16_noties_2507.json evaluation_classification_final.json -nt -p

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_zero_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few2_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few4_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few8_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few16_noties_1007.json evaluation_classification.json -nt -cc -p
# 
# # on kept samples
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few16_1007.json evaluation_classification.json -p
# 
# # on discarded samples
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/GPT/predictions_gpt4.1_few16_1007.json evaluation_classification.json -p
# 
# # on 1806 CROSS VALIDATION
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1_zero_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1_zero_3005.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1_few_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1_few_3005.json evaluation_classification_108.json -nt -cc -p
# 
# # noties

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1mini_zero_noties_1806_2507.json,predictions/classification/108_cross_validation/GPT/predictions_gpt4.1mini_zero_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_gpt4.1mini_few_noties_1806_2507.json,predictions/classification/108_cross_validation/GPT/predictions_gpt4.1mini_few_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv

# ## Deepseek
# 
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_zero_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few2_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few4_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few8_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few16_1007.json evaluation_classification.json -nt -cc -p
# 
# # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_zero_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few2_noties_2507.json evaluation_classification_final.json -nt -p -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few4_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few8_noties_2507.json evaluation_classification_final.json -nt -p -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few16_noties_2507.json evaluation_classification_final.json -nt -p

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_zero_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few2_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few4_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few8_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few16_noties_1007.json evaluation_classification.json -nt -cc -p
# 
# # on kept samples
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few16_1007.json evaluation_classification.json -p
# 
# # on discarded samples
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/deepseek/predictions_deepseek_few16_1007.json evaluation_classification.json -p
# 
# # on 1806 CROSS VALIDATION
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_zero_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_zero_3005.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_few_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_few_3005.json evaluation_classification_108.json -nt -cc -p
# 
# # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/deepseek/predictions_deepseek_zero_noties_1806_2507.json,predictions/classification/108_cross_validation/deepseek/predictions_deepseek_zero_noties_3005_2507.json evaluation_classification_108_final.json -nt -p -cv
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/deepseek/predictions_deepseek_few_noties_1806_2507.json,predictions/classification/108_cross_validation/deepseek/predictions_deepseek_few_noties_3005_2507.json evaluation_classification_108_final.json -nt -p -cv

# ## Mistral
# 
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_zero_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few2_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few4_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few8_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few16_1007.json evaluation_classification.json -nt -cc -p
# 
# # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_zero_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few2_noties_2507.json evaluation_classification_final.json -nt -p -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few4_noties_2507.json evaluation_classification_final.json -nt -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few8_noties_2507.json evaluation_classification_final.json -nt -p -m mistakes.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few16_noties_2507.json evaluation_classification_final.json -nt -p

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_zero_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few2_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few4_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few8_noties_1007.json evaluation_classification.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few16_noties_1007.json evaluation_classification.json -nt -cc -p
# 
# # on kept samples
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_kept_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few16_1007.json evaluation_classification.json -p
# 
# # on discarded samples
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_zero_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few2_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few4_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few8_1007.json evaluation_classification.json -p
# python3 evaluate_predictions.py gold_standards/gold_standard_discarded_items_1806.json predictions/classification/200_held_out/mistral/predictions_mistral7B_few16_1007.json evaluation_classification.json -p

# # on 1806 CROSS VALIDATION
# # with ties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_zero_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_zero_3005.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_few_1806.json evaluation_classification_108.json -nt -cc -p
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/GPT/predictions_deepseek_few_3005.json evaluation_classification_108.json -nt -cc -p
# 
# # noties
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/mistral/predictions_mistral_zero_noties_1806_2507.json,predictions/classification/108_cross_validation/mistral/predictions_mistral_zero_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/classification/108_cross_validation/mistral/predictions_mistral_few_noties_1806_2507.json,predictions/classification/108_cross_validation/mistral/predictions_mistral_few_noties_3005_2507.json evaluation_classification_108_final.json -nt -cc -p -cv

# # NLI
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_bart_1207_c.json evaluation_nli.json -nli -nt -p -m mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_roberta_1207_c.json evaluation_nli.json -nli -nt -p -m mistakes_nli.json
# 
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_bart_1207_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_roberta_1207_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes_nli.json

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_bart_2107_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes_finetuned_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_roberta_2107_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes_finetuned_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_deberta_2107_c.json evaluation_nli.json -nli -nt -p -m mistakes_finetuned_nli.json

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_bart_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes/mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_impl_roberta_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes/mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_impl_deberta_c.json evaluation_nli.json -nli -nt -p -m mistakes/mistakes_nli.json

# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_pre_bart_optimum_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes/mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_pre_quest_roberta_c.json evaluation_nli.json -nli -nt -p -sp 0.0 -m mistakes/mistakes_nli.json
# python3 evaluate_predictions.py gold_standards/gold_standard_merged.json predictions/nli/predictions_mergedstudy_finetuned_imppres_pre_deberta_c.json evaluation_nli.json -nli -nt -p -m mistakes/mistakes_nli.json