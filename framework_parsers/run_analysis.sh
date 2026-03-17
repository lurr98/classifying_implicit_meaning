#!/bin/bash

# python3 ./analyse_frameworks.py "constituency/constituency_parsed.json,framenet/framenet_parsed.json" "../../implicit_data/gold_standard.json" "constituency_framenet_overlap_analysis.json" -a -l No,Yes
# python3 ./analyse_frameworks.py "constituency/constituency_parsed.json,rst/rst_parsed.json" "../../implicit_data/gold_standard.json" "constituency_rst_overlap_analysis.json" -a -l No,Yes
# python3 ./analyse_frameworks.py "rst/rst_parsed.json,framenet/framenet_parsed.json,constituency/constituency_parsed.json" "../../implicit_data/gold_standard.json" "all_parsers_overlap_analysis.json" -a -l No,Yes

# python3 ./analyse_frameworks.py "../../analysis/frameworks/all_parsers_overlap_analysis.json" "all_parsers_overlap.png" -p -l No,Yes
python3 ./analyse_frameworks.py "../../analysis/frameworks/overlap_analysis.json" "rst_framenet_overlap.png" -p -l No,Yes

exit 1
# python3 ./analyse_frameworks.py "./constituency/constituency_parsed_smerged.json" "../../implicit_data/gold_standard.json" "constituency_analysis_smerged.json" -a -l No,Yes
# python3 ./analyse_frameworks.py "../../analysis/frameworks/constituency_analysis_smerged.json" "constituency_relations_smerged_counts.png" -p -l No,Yes
# python3 ./analyse_frameworks.py "../../analysis/frameworks/rst_analysis.json" "rst_relations_counts.png" -p -l No,Yes
# python3 ./analyse_frameworks.py "../../analysis/frameworks/framenet_analysis.json" "framenet_relations_counts.png" -p -l No,Yes

topics=("arts" "business" "cars" "computer" "education" "family" "food" "garden" "health" "hobbies" "pets" "philosophy" "relationship" "sports" "style" "travel" "work" "youth")
splits=("train" "dev" "test" "out_of_domain_test")

# for topic in "${topics[@]}"; do
for split in "${splits[@]}"; do

    # python3 ./analyse_frameworks.py "./framenet/framenet_parsed.json" "../../implicit_data/gold_standard.json" "${topic}_framenet_analysis.json" -a -l No,Yes -t $topic
    # python3 ./analyse_frameworks.py "./framenet/rst_parsed.json" "../../implicit_data/gold_standard.json" "${topic}_rst_analysis.json" -a -l No,Yes -t $topic

    # python3 ./analyse_frameworks.py "./framenet/framenet_parsed.json" "../../implicit_data/gold_standard.json" "${split}_framenet_analysis.json" -a -l No,Yes -s $split
    # python3 ./analyse_frameworks.py "./rst/rst_parsed.json" "../../implicit_data/gold_standard.json" "${split}_rst_analysis.json" -a -l No,Yes -s $split
    # python3 ./analyse_frameworks.py "./constituency/constituency_parsed_smerged.json" "../../implicit_data/gold_standard.json" "${split}_constituency_analysis_smerged.json" -a -l No,Yes -s $split

    # python3 ./analyse_frameworks.py "../../analysis/frameworks/${topic}_framenet_analysis.json" "${topic}_framenet_relations.png" -p -l No,Yes -t $topic -perc
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/${topic}_rst_analysis.json" "${topic}_rst_relations.png" -p -l No,Yes -t $topic -perc

    python3 ./analyse_frameworks.py "../../analysis/frameworks/${split}_framenet_analysis.json" "${split}_framenet_relations_counts.png" -p -l No,Yes -s $split
    python3 ./analyse_frameworks.py "../../analysis/frameworks/${split}_rst_analysis.json" "${split}_rst_relations_counts.png" -p -l No,Yes -s $split
    # python3 ./analyse_frameworks.py "../../analysis/frameworks/${split}_constituency_analysis_smerged.json" "${split}_constituency_relations_smerged_counts.png" -p -l No,Yes -s $split
done