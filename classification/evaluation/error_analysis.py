import json, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


def load_data(path):
    with open(path) as f:
        return json.load(f)

def flatten_errors(error_dict):
    """Merge all settings into a single error set."""
    ids = set()
    for ids_list in error_dict.values():
        ids.update(ids_list)
    return ids


def error_overlap(models_errors):
    models_old_names = sorted(models_errors.keys())
    models = [plot_names[m] for m in sorted(models_errors.keys())]

    mat = pd.DataFrame(
        np.zeros((len(models), len(models))),
        index=models,
        columns=models
    )

    # diagonal = 1 (100% of a model's errors overlap with itself)
    for m in models:
        mat.loc[m, m] = 1.0

    for m1, m2 in combinations(models_old_names, 2):
        e1 = models_errors[m1]
        e2 = models_errors[m2]
        shared = len(e1 & e2)

        if len(e1) > 0:
            mat.loc[plot_names[m1], plot_names[m2]] = shared / len(e1)
        if len(e2) > 0:
            mat.loc[plot_names[m2], plot_names[m1]] = shared / len(e2)

    return mat


def plot_error_overlap_heatmap(overlap_matrix, k, split, title=None):
    plt.figure(figsize=(8, 7))

    sns.heatmap(
        overlap_matrix,
        cmap="Reds",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Proportion of shared errors"},
        annot=True,
        fmt=".2f"
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    if title:
        plt.title(title, pad=12)

    plt.tight_layout()
    plt.savefig(f"mistakes_{split}_{k}N.png", bbox_inches='tight')


def common_group_errors(models_errors, model_groups, group_name):
    models = model_groups[group_name]
    error_sets = [
        models_errors[m] for m in models if m in models_errors
    ]

    if not error_sets:
        return set()

    intersection = set.intersection(*error_sets)

    return intersection


def exclusive_group_errors(group_errors, target_group):
    target = group_errors[target_group]
    print(target)
    others = set().union(
        *[v for k, v in group_errors.items() if k != target_group]
    )
    return target - others


def compute_error_frequencies(models_errors):
    """
    models_errors: dict[model_name -> set(error_ids)]
    """
    counter = Counter()
    for error_set in models_errors.values():
        counter.update(error_set)
    return counter


def get_hard_items(counter, min_models=3):
    """
    Return items misclassified by at least min_models models.
    """
    return {
        item: freq
        for item, freq in counter.items()
        if freq >= min_models
    }


def index_linguistic_data(ling_data):
    indexed = {}
    for i, item_id in enumerate(ling_data["IDs"]):
        indexed[item_id] = {
            "target": ling_data["targets"][i],
            "text_length": ling_data["text_length"][i],
            "av_word_length": ling_data["av_word_length"][i],
            "lexical_density": ling_data["lexical_density"][i],
            "POS_distr": ling_data["POS_distr"][i]
        }
    return indexed


def summarize_linguistics(item_ids, ling_index):
    feats = {
        "text_length": [],
        "av_word_length": [],
        "lexical_density": [],
        "POS_distr": []
    }

    for i in item_ids:
        if i not in ling_index:
            continue
        d = ling_index[i]
        feats["text_length"].append(d["text_length"])
        feats["av_word_length"].append(d["av_word_length"])
        feats["lexical_density"].append(d["lexical_density"])
        feats["POS_distr"].append(d["POS_distr"])

    return {
        "text_length": np.mean(feats["text_length"]),
        "av_word_length": np.mean(feats["av_word_length"]),
        "lexical_density": np.mean(feats["lexical_density"]),
        "POS_distr": average_pos(feats["POS_distr"])
    }

def average_pos(pos_dicts):
    total = Counter()
    for d in pos_dicts:
        total.update(d)
    s = sum(total.values())
    return {k: v / s for k, v in total.items()}


def compare_hard_easy(hard_items, all_ids, ling_index):
    hard_ids = set(hard_items)
    easy_ids = set(all_ids) - hard_ids

    return {
        "hard": summarize_linguistics(hard_ids, ling_index),
        "easy": summarize_linguistics(easy_ids, ling_index)
    }

def get_mistake_topics(id2topic: dict, mistake_ids: set, slplit_ids: list):
    topics_mistakes = [id2topic[i] for i in mistake_ids if i in id2topic]
    topics_overall = [id2topic[i] for i in split_ids if i in id2topic]
    counter_mistakes = Counter(topics_mistakes)
    counter_overall = Counter(topics_overall)
    mistake_dist = {k: v for k, v in counter_mistakes.items()}
    for k,v in mistake_dist.items():
        mistake_dist[k] = {
            "count": v,
            "percentage to topic distribution": v / counter_overall[k],
            "percentage to all mistakes": v / len(mistake_ids)
        }
    return mistake_dist


def get_label_distribution(golds: dict, mistake_ids: set, split_ids: list):
    labels_mistakes = [str(tuple(golds[i]["Label Distribution"])) for i in mistake_ids if i in golds]
    labels_overall = [str(tuple(golds[i]["Label Distribution"])) for i in split_ids if i in golds]
    counter_mistakes = Counter(labels_mistakes)
    counter_overall = Counter(labels_overall)
    mistake_dist = {k: v for k, v in counter_mistakes.items()}
    for k,v in mistake_dist.items():
        mistake_dist[k] = {
            "count": v,
            "percentage to label distribution": v / counter_overall[k],
            "percentage to all mistakes": v / len(mistake_ids)
        }
    return mistake_dist


def get_mistakes_relations(mistake_ids: set, rst_rels: dict, fn_rels: dict):
    rst_mistakes = [rst_rels[i]["relevant_segment"]["relname"] for i in mistake_ids if i in rst_rels and "relevant_segment" in rst_rels[i]]
    fn_mistakes = [fn_rels[i]["relevant_frame"]["element"] for i in mistake_ids if i in fn_rels and "relevant_frame" in fn_rels[i]]
    counter_rst = Counter(rst_mistakes)
    counter_fn = Counter(fn_mistakes)
    return {
        "RST_relations": {k: v for k, v in counter_rst.items()},
        "FrameNet_relations": {k: v for k, v in counter_fn.items()}
    }

if __name__ == "__main__":

    ling_file = "/anvme/workspace/v106be21-arr_workspace_december/analysis/metric_dicts/mace_labels/this/metric_dict.json"
    ling_data = load_data(ling_file)

    id2topic = "/anvme/workspace/v106be21-arr_workspace_december/implicit_data/id2topic.json"
    topic_data = load_data(id2topic)

    gold_file = "/anvme/workspace/v106be21-arr_workspace_december/implicit_data/gold_standard.json"
    golds = load_data(gold_file)

    split_file = "/anvme/workspace/v106be21-arr_workspace_december/implicit_data/splits.json"
    split_data = load_data(split_file)

    rst_path = "/anvme/workspace/v106be21-arr_workspace_december/classifying_implicit_meaning/framework_parsers/rst/rst_parsed.json"
    fn_path = "/anvme/workspace/v106be21-arr_workspace_december/classifying_implicit_meaning/framework_parsers/framenet/framenet_parsed.json"

    rst_rels = load_data(rst_path)
    fn_rels = load_data(fn_path)

    base_dir = "/anvme/workspace/v106be21-arr_workspace_december/evaluation"

    models = ["gpt", 
                "deepseek", 
                "qwen",
                "llama",
                "mistral",
                "bart",
                "roberta",
                "deberta"
            ]
    # groups = {"LLMs": ["gpt", "mistral", "llama", "qwen"], "NLI": ["roberta", "deberta", "bart"]}
    groups = {
        "ALL": ["gpt", "deepseek", "mistral", "llama", "qwen", "roberta", "deberta", "bart"], 
        "LLMs": ["gpt", "deepseek", "mistral", "llama", "qwen"], 
        "NLI": ["roberta", "deberta", "bart"],
        "GD": ["gpt", "deepseek"],
        "Mixtral": ["mistral", "llama", "qwen"]}

    plot_names = {
        "test": "Test Split", 
        "out_of_domain_test": "Out-of-Domain Test Split",
        "gpt": "GPT",
        "deepseek": "DeepSeek",
        "mistral": "Mistral",
        "llama": "LLaMA",
        "qwen": "Qwen",
        "roberta": "RoBERTa",
        "deberta": "DeBERTa",
        "bart": "BART"}

    # key = {"classification": "zero", "nli": "c_secondprob0"}
    # key = {"classification": "zero", "nli": "zero"}
    key = {"classification": "best_ft", "nli": "best_ft"}
    models_errors = {}
    for split in ["test", "out_of_domain_test"]:
        print(f"\n--- Analyzing {split} split ---\n")
        split_ids = split_data[split]
        for model_class in ["classification", "nli"]:
            mistake_path = os.path.join(base_dir, model_class, split, "mistakes")

            for file in os.listdir(mistake_path):
                if not file.endswith(".json") or file.split("_")[-1][:-5] not in models:
                    continue
                print(file)
                path = os.path.join(mistake_path, file)
                mistakes = load_data(path)
                # flat_errors = flatten_errors(mistakes)
                # print(f"{file}: {len(flat_errors)} errors")
                print(path)
                if key[model_class] in mistakes:
                    models_errors[file.split("_")[-1][:-5]] = set(mistakes[key[model_class]])
                elif f"{key[model_class]}_updintro" in mistakes: 
                    models_errors[file.split("_")[-1][:-5]] = set(mistakes[f"{key[model_class]}_updintro"])

            print(f"Collected errors for models: {models_errors}")

        print(f"\nError overlaps for {split} split:")
        overlaps = error_overlap(models_errors)
        plot_error_overlap_heatmap(overlaps, key["classification"], split,title=f"Error Overlap Heatmap - {plot_names[split]}")
        for model_pair, stats in overlaps.items():
            print(f"{model_pair}: {stats}")

        hard_items = get_hard_items(compute_error_frequencies(models_errors), min_models=3)
        print(f"\nNumber of hard items (misclassified by at least 3 models OVERALL): {len(hard_items)}")

        compare_hard_easy_results = compare_hard_easy(
            hard_items,
            ling_index=index_linguistic_data(ling_data),
            all_ids=ling_data["IDs"]
        )
        print(f"\nLinguistic comparison of hard vs easy items:")
        print(json.dumps(compare_hard_easy_results, indent=2))

        group_errors = {g: {m: errors for m, errors in models_errors.items() if m in groups[g]} for g in groups}
        common_errors = {g: common_group_errors(models_errors, groups, g) for g in groups}
        exclusive_errors = {group: exclusive_group_errors(common_errors, group) for group in groups}
        for group in groups:
            print(f"{group} common errors: {len(common_errors[group])}")
            print(f"Common {group} errors: {common_errors[group]}")

            print(f"{group} exclusive errors: {len(exclusive_errors[group])}")
            print(f"Exclusive {group} errors: {exclusive_errors[group]}")

            print(f"{group} label distribution of mistakes:")
            label_dist = get_label_distribution(golds, common_errors[group], split_ids)
            print(json.dumps(label_dist, indent=2))

            hard_items_group = get_hard_items(compute_error_frequencies(group_errors[group]), min_models=3)
            print(f"{group} hard items (misclassified by at least 3 models in group): {len(hard_items_group)}")

            topic_dist = get_mistake_topics(topic_data, common_errors[group], split_ids)
            print(f"{group} mistake topic distribution:")
            print(json.dumps(topic_dist, indent=2))

            fw_relations_group = get_mistakes_relations(common_errors[group], rst_rels, fn_rels)
            print(f"{group} mistake framework relations:")
            print(json.dumps(fw_relations_group, indent=2))
    