#!/bin/bash

# models=("gpt" "deepseek" "llama" "mistral" "qwen")
models=("gpt" "deepseek" "llama" "mistral" "qwen" "bart" "roberta" "deberta")
parsers=("constituency" "rst" "framenet")

# first run: test
touch "all_model_mistakes_overlap.json"
for parser in "${parsers[@]}"; do
    touch "all_model_mistakes_${parser}.json"
done

for model in "${models[@]}"; do

    # python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/overlap.png" -p -l No,Yes -m ../../evaluation/classification/test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/overlap_nli.png" -p -l No,Yes -m ../../evaluation/nli/test/mistakes/mistakes_${model}.json -perc

    # # Constituency
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/test/constituency_phrases_smerged.png" -p -l No,Yes -m ../../evaluation/classification/test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/test/constituency_phrases_nli_smerged.png" -p -l No,Yes -m ../../evaluation/nli/test/mistakes/mistakes_${model}.json -perc

    # # RST
    python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations.png" -p -l No,Yes -m ../../evaluation/classification/test/mistakes/mistakes_${model}.json -perc
    python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations_nli.png" -p -l No,Yes -m ../../evaluation/nli/test/mistakes/mistakes_${model}.json -perc
    # 
    # # FrameNet
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements.png" -p -l No,Yes -m ../../evaluation/classification/test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements_nli.png" -p -l No,Yes -m ../../evaluation/nli/test/mistakes/mistakes_${model}.json -perc
done

# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/all_models_overlap.png" -p -l No,Yes -m all_model_mistakes_overlap.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/all_models_overlap.png" -p -l No,Yes -m all_model_mistakes_overlap.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/all_models_overlap_nli.png" -p -l No,Yes -m all_model_mistakes_overlap.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/test/all_models_overlap_nli.png" -p -l No,Yes -m all_model_mistakes_overlap.json

# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/test/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/test/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis.json" "mistakes/test/constituency_phrases_nli.png" -p -l No,Yes -m all_model_mistakes_constituency.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis.json" "mistakes/test/constituency_phrases_nli.png" -p -l No,Yes -m all_model_mistakes_constituency.json
# 
# python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations.png" -p -l No,Yes -m all_model_mistakes_rst.json -perc
python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations.png" -p -l No,Yes -m all_model_mistakes_rst.json
# # python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations_nli.png" -p -l No,Yes -m all_model_mistakes_rst.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/test/rst_relations_nli.png" -p -l No,Yes -m all_model_mistakes_rst.json
# # 
# # # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json -perc
# # # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/test/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json

exit 1
for parser in "${parsers[@]}"; do
    # rm "all_model_mistakes_${parser}.json"
    touch "all_model_mistakes_ood_${parser}.json"
done
rm "all_model_mistakes_overlap.json"
touch "all_model_mistakes_overlap.json"

# second run: out_of_domain
for model in "${models[@]}"; do

    # python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/overlap.png" -p -l No,Yes -m ../../evaluation/classification/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/overlap_nli.png" -p -l No,Yes -m ../../evaluation/nli/out_of_domain_test/mistakes/mistakes_${model}.json -perc

    # Constituency
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged.png" -p -l No,Yes -m ../../evaluation/classification/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged_nli.png" -p -l No,Yes -m ../../evaluation/nli/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    #
    # # RST
    python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations.png" -p -l No,Yes -m ../../evaluation/classification/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations_nli.png" -p -l No,Yes -m ../../evaluation/nli/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    #  
    # # FrameNet
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements.png" -p -l No,Yes -m ../../evaluation/classification/out_of_domain_test/mistakes/mistakes_${model}.json -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements_nli.png" -p -l No,Yes -m ../../evaluation/nli/out_of_domain_test/mistakes/mistakes_${model}.json -perc
done

# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/all_models_overlap.png" -p -l No,Yes -m all_model_mistakes_overlap.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/all_models_overlap.png" -p -l No,Yes -m all_model_mistakes_overlap.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/all_models_overlap_nli.png" -p -l No,Yes -m all_model_mistakes_overlap.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "mistakes/out_of_domain/all_models_overlap_nli.png" -p -l No,Yes -m all_model_mistakes_overlap.json


# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "mistakes/out_of_domain/constituency_phrases_smerged.png" -p -l No,Yes -m all_model_mistakes_constituency.json
# 
# # python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations.png" -p -l No,Yes -m all_model_mistakes_rst.json -perc
# # python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations.png" -p -l No,Yes -m all_model_mistakes_rst.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations.png" -p -l No,Yes -m all_model_mistakes_rst.json -perc
python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "mistakes/out_of_domain/rst_relations.png" -p -l No,Yes -m all_model_mistakes_ood_rst.json
# # 
# # # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json -perc
# # # python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json
# python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json -perc
# python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "mistakes/out_of_domain/framenet_elements.png" -p -l No,Yes -m all_model_mistakes_framenet.json

# for parser in "${parsers[@]}"; do
#     rm "all_model_mistakes_${parser}.json"
# done
rm "all_model_mistakes_overlap.json"
