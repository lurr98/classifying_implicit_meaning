import json, random, re, spacy, pickle, os
import pandas as pd
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset
from typing import Tuple, Union
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


selected_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET"]
nlp = spacy.load("en_core_web_sm")


def read_json(json_file: str) -> Union[dict, list]:

    with open(json_file, "r") as jsn:
        loaded_dict = json.load(jsn)

    return loaded_dict
    

def write_json(json_dict: dict, json_file: str) -> None:

    with open(json_file, "w") as jsn:
        json.dump(json_dict, jsn)


def append_json(json_dict: dict, json_file: str) -> None:
    
    data = {}

    # load existing data if file exists and is not empty
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)   # expects a dict

    # update dict
    for key, value in json_dict.items():
        if key in data:
            for k, v in value.items():
                data[key][k] = v 
        else:
            data[key] = value

    # write back full JSON object
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_revision_label(gold_standard: dict) -> list:

    rev_list = []
    for id, sample in gold_standard.items():
        rev = re.findall(r"<(.*)>", sample["Revised"])
        rev_list.append((id, rev[0], sample["Majority Vote"]))

    return rev_list


def merge_data(data1: dict, data2: dict, out: str) -> None:

    with open(data1, "r") as d1:
        data1_loaded = json.load(d1)

    with open(data2, "r") as d2:
        data2_loaded = json.load(d2)

    merged_dict = {**data1_loaded, **data2_loaded}

    with open(out, "w") as o:
        json.dump(merged_dict, o)


def filter_ties(gold_standard: dict) -> dict:

    filtered_dict = {}
    for idx, sample in gold_standard.items():
        if sample["Majority Vote"] != "Tie":
            filtered_dict[idx] = sample
    return filtered_dict


def prepare_few_shot(samples: dict, few_numbers: list, labels: list, preferred_labels: list, balance: bool=True) -> Tuple[dict, dict]:

    def balancing_procedure(dyn_label_pools: dict, base_count: int, remainder: int, collection: list):

        this_few_dict = {}
        for i, pool in dyn_label_pools.items():
            this_few_dict[i] = {}
            # sample base counts from all labels
            for label, label_samples in pool.items():
                this_few_dict[i][label] = []
                # print(f"Base count: {random.sample(label_samples, base_count)}")
                rand_samples = random.sample(label_samples, base_count)
                this_few_dict[i][label].extend(rand_samples)
                
                for rand in rand_samples:
                    if rand not in collection:
                        print(rand["ID"])
                        collection.append(rand)
            random.shuffle(preferred_labels)
            # take remainder from preferred labels
            # TODO: change for case that preferred labels are not n-1 
            for j in list(range(remainder)):
                rand_sample = None
                while rand_sample is None:
                    rand_sample = random.sample(pool[preferred_labels[j]], 1)
                    if rand_sample[0] in this_few_dict[i][preferred_labels[j]]:
                        rand_sample = None
                        
                this_few_dict[i][preferred_labels[j]].extend(rand_sample)
                print("Preferred")
                print(this_few_dict[i][preferred_labels[j]])
                if rand_sample[0] not in collection:
                    print(rand_sample[0]["ID"])
                    print()
                    collection.extend(rand_sample)

        return this_few_dict, collection

    highest_few = max(few_numbers)

    few_dict = {}
    if balance:
        data_mixes = list(set([k[2] for k, sample in samples.items()]))
        print(data_mixes)
        data_mix_ratio = highest_few / len(data_mixes)
        pools = [[{**sample, **{"ID": k}} for k, sample in samples.items() if k[2] == i] for i in data_mixes]
        # shape [number of data mixes, number of labels]
        label_pools = {i: {label: [sample for sample in pool if sample["Majority Vote"] == label] for label in labels} for i, pool in enumerate(pools)}
        
        print(label_pools)
        # few_dict[highest_few] = balancing_procedure(label_pools, base_count, remainder)
        dynamic_labels_pools = label_pools

        collection = []
        # go through all number but the highest one (we already took care of that one)
        for num in sorted(few_numbers, reverse=True):
            # divide by number of data mixes
            pool_num = num / len(pools)
            # get the base count for every class
            base_count = int(pool_num // len(labels))
            # get remaining slots
            remainder = int(pool_num % len(labels))
            few_dict[num], collection = balancing_procedure(dynamic_labels_pools, base_count, remainder, collection)
            dynamic_labels_pools = few_dict[num]
    
    return few_dict, collection


def select_balanced_few_shots(gold_standard: dict,
                              few_numbers: list = [2, 4, 8, 16],
                              labels: list = ["No", "Yes", "Tie"],
                              preferred_labels: list = ["No", "Yes"],
                              seed: int = 42) -> Tuple[dict, list]:
    """
    Deterministically select label-balanced few-shot samples from a gold standard dict.
    Samples are CUMULATIVE: all samples in size N are included in size N+M.

    Args:
        gold_standard: mapping id -> sample dict (expects key "Majority Vote").
        few_numbers: list of sizes to produce (e.g. [2,4,8,16]).
        labels: list of labels to balance across (order matters for base allocation).
        preferred_labels: order to draw remainder samples from when num % len(labels) != 0.
        seed: base random seed for deterministic selection.

    Returns:
        few_shots_organized: dict mapping each requested size -> {"0": {label: [samples...]}}
        collection: flattened list of all selected sample dicts (each includes an "ID" field).

    Notes:
        - Selection is deterministic given the same seed.
        - Smaller few-shot sets are subsets of larger ones (cumulative).
    """

    import random

    # Build pools per label
    label_pools = {lab: [{"ID": idx, **sample} for idx, sample in gold_standard.items() if sample.get("Majority Vote") == lab]
                   for lab in labels}

    few_shots_organized = {}
    cumulative_selected = {lab: [] for lab in labels}  # Track all selections cumulatively
    
    # Process sizes in ascending order so smaller sets are subsets of larger ones
    for num in sorted(few_numbers):
        rnd = random.Random(seed + num)
        base_count = num // len(labels)
        remainder = num % len(labels)

        selected_by_label = {lab: [] for lab in labels}

        # For each label, select base_count samples (preferring those not yet selected)
        for lab in labels:
            pool = label_pools.get(lab, [])
            already_selected = set(s["ID"] for s in cumulative_selected[lab])
            
            # First, keep all previously selected samples
            selected_by_label[lab].extend(cumulative_selected[lab])
            
            # Then add new samples to reach base_count
            new_needed = base_count - len(selected_by_label[lab])
            if new_needed > 0:
                available = [p for p in pool if p["ID"] not in already_selected]
                if len(available) <= new_needed:
                    selected_by_label[lab].extend(available)
                else:
                    new_samples = rnd.sample(available, new_needed)
                    selected_by_label[lab].extend(new_samples)

        # Fill remainder from preferred labels
        prefs = [p for p in preferred_labels if p in labels]
        for i in range(remainder):
            if not prefs:
                break
            lab = prefs[i % len(prefs)]
            already_selected = set(s["ID"] for s in cumulative_selected[lab])
            available = [p for p in label_pools.get(lab, []) if p["ID"] not in already_selected and p not in selected_by_label[lab]]
            if not available:
                # Try any other label with available samples
                for other in labels:
                    already_selected_other = set(s["ID"] for s in cumulative_selected[other])
                    available = [p for p in label_pools.get(other, []) if p["ID"] not in already_selected_other and p not in selected_by_label[other]]
                    if available:
                        lab = other
                        break
            if available:
                chosen = rnd.choice(available)
                selected_by_label[lab].append(chosen)

        # Update cumulative tracking
        for lab in labels:
            cumulative_selected[lab] = selected_by_label[lab]

        few_shots_organized[num] = {"0": selected_by_label}

    # Build final collection from largest set (contains all samples)
    collection = []
    if few_numbers:
        largest_num = max(few_numbers)
        for lab, samples in few_shots_organized[largest_num]["0"].items():
            for s in samples:
                if s not in collection:
                    collection.append(s)

    return few_shots_organized, collection


## TODO: check these functions!!!!!!
def format_few_examples(few_examples_loaded: dict, samples_file_loaded: dict, introduction: str, num: int):

    few_samples, gold_labels = [], []
    for label, samples in few_examples_loaded[str(num)].items():
        few_samples.extend(samples)
        gold_labels.extend([label for _ in range(len(samples))])

    temp = list(zip(few_samples, gold_labels))  # Pair the elements
    random.shuffle(temp)  # Shuffle the pairs
    few_samples, gold_labels = zip(*temp)  # Unzip into separate lists
    sample_str = ""

    for i, item in enumerate(few_samples):
        print(item)
        id_sample = samples_file_loaded[item]
        sample_str += f"Example {i+1}:\nArticle Name: {id_sample['article_name']}\nFirst Text:\n{id_sample['context_before']} **{id_sample['sentence_1']}** {id_sample['context_after']}\nSecond Text:\n{id_sample['context_before']} **{id_sample['sentence_2']}** {id_sample['context_after']}\nLabel: {gold_labels[i]}\n\n"

    return introduction + sample_str


def format_target_sample(item: dict) -> str:

    formatted_item = f"*Article name:* {item['article_name']}\n\
        Read the following text and focus on the **bold sentence**.\n\
        First Text:\n{item['context_before']} **{item['sentence_1']}** {item['context_after']}\n\n\
        Now read the modified text:\n\
        Second Text:\n{item['context_before']} **{item['sentence_2']}** {item['context_after']}\n\nWhat would most readers say?\nWould altering the bold sentence meaningfully change how they understand the text?"
    
    return formatted_item


def separate_test_data(all_samples: dict, few_examples: dict):

    ids = set([few["ID"] for few in few_examples])

    test_data = {}

    for idx, sample in all_samples.items():
        if not idx in ids:
            test_data[idx] = sample

    return test_data


# FINE-TUNING

def load_impres_data() -> dict:

    train_val_dict = {"train": None, "val": None}
    # available_configs = ['implicature_connectives', 'implicature_gradable_adjective', 'implicature_gradable_verb', 'implicature_modals', 'implicature_numerals_10_100', 'implicature_numerals_2_3', 'implicature_quantifiers', 'presupposition_all_n_presupposition', 'presupposition_both_presupposition', 'presupposition_change_of_state', 'presupposition_cleft_existence', 'presupposition_cleft_uniqueness', 'presupposition_only_presupposition', 'presupposition_possessed_definites_existence', 'presupposition_possessed_definites_uniqueness', 'presupposition_question_presupposition']
    available_configs = ['implicature_connectives', 'implicature_gradable_adjective', 'implicature_gradable_verb', 'implicature_modals', 'implicature_numerals_10_100', 'implicature_numerals_2_3', 'implicature_quantifiers']
    # available_configs = ['presupposition_all_n_presupposition', 'presupposition_both_presupposition', 'presupposition_change_of_state', 'presupposition_cleft_existence', 'presupposition_cleft_uniqueness', 'presupposition_only_presupposition', 'presupposition_possessed_definites_existence', 'presupposition_possessed_definites_uniqueness', 'presupposition_question_presupposition']
    # # OPTIMUM for BART
    # available_configs = ['presupposition_change_of_state', 'presupposition_cleft_existence', 'presupposition_cleft_uniqueness', 'presupposition_only_presupposition', 'presupposition_possessed_definites_existence', 'presupposition_possessed_definites_uniqueness', 'presupposition_question_presupposition']
    # # OPTIMUM for RoBERTa
    # available_configs = ['presupposition_change_of_state', 'presupposition_cleft_existence', 'presupposition_only_presupposition', 'presupposition_possessed_definites_existence', 'presupposition_possessed_definites_uniqueness', 'presupposition_question_presupposition']
    # OPTIMUM for DeBERTa
    # available_configs = ['presupposition_change_of_state', 'presupposition_possessed_definites_existence', 'presupposition_possessed_definites_uniqueness']

    imppres_mapping = {0: 2, 2: 0, 1: 1}
    # imppres_mapping = {0: 0, 2: 2, 1: 1}
    pd_dict = {"premise": [], "hypothesis": [], "label": []}
    for config in available_configs:
        ds = load_dataset("facebook/imppres", config)
        ds_pand = ds["_".join(config.split("_")[1:])].to_pandas()
        pd_dict["premise"].extend([x for x in ds_pand["premise"]])
        pd_dict["hypothesis"].extend([x for x in ds_pand["hypothesis"]])
        if "gold_label" in ds_pand:
            pd_dict["label"].extend([imppres_mapping[x] for x in ds_pand["gold_label"]])
        else:
            try:
                pd_dict["label"].extend([imppres_mapping[x] for x in ds_pand["gold_label_prag"]])
            except ValueError:
                print(ds_pand.columns)

    counter = Counter(pd_dict["label"])
    for label in set(pd_dict["label"]):
        print(f"{label}: {counter[label]} counts")
    exit()
    df = pd.DataFrame(pd_dict)
    df = df.sample(frac=1).reset_index(drop=True)

    val_df = df.sample(frac=1/10, random_state=42)
    train_df = df.drop(index=val_df.index)

    train_val_dict["train"] = train_df
    train_val_dict["val"] = val_df

    return train_val_dict


def transform_inli_data(inli_files: list) -> dict:

    train_val_dict = {"train": None, "val": None}
    for file in inli_files:
        with open(file, "r") as inf:
            inli_lines = inf.readlines()

        pd_dict = {"premise": [], "hypothesis": [], "label": []}
        inli_mapping = {"neutral": 1, "contradiction": 0, "implied_entailment": 2, "explicit_entailment": 2}

        for line in inli_lines[1:]:
            for i in list(range(3,7)):
                pd_dict["premise"].append(line.split(",")[2])
                pd_dict["hypothesis"].append(line.split(",")[i])
                pd_dict["label"].append(inli_mapping[inli_lines[0].split(",")[i].strip()])
        
        df = pd.DataFrame(pd_dict)
        df = df.sample(frac=1).reset_index(drop=True)
        if "train" in file:
            train_val_dict["train"] = df
        else:
            train_val_dict["val"] = df
    return train_val_dict


def pos_tagger(texts: list) -> list:

    nlp = spacy.load("en_core_web_sm") 
    pos_tags = []
    for text in texts:
        doc = nlp(text)
        pos_tags.append(" ".join([t.pos_ for t in doc]))

    return pos_tags


## LINGUISTICS

def compute_lexical_density(revision: str):
    doc = nlp(revision)
    total_tokens = len([token for token in doc if not token.is_punct and not token.is_space])
    
    content_words = [
        token for token in doc
        if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}
        and not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]
    
    density = len(content_words) / total_tokens if total_tokens > 0 else 0
    return round(density, 3)


def average_categories(selected_categories: list, metric: str, metric_dict: dict) -> dict:

    cat_dict = {label: [0 for cat in selected_categories] for label in list(metric_dict.keys())}

    for label in list(metric_dict.keys()):
        for sample_distr in metric_dict[label][metric]:
            for pos_collected, distr in sample_distr.items():
                if pos_collected in selected_categories:
                    cat_dict[label][selected_categories.index(pos_collected)] += distr

        total = len(metric_dict[label][metric])
        cat_dict[label] = [item / total for item in cat_dict[label]]

    return cat_dict


def get_metrics(rev_list: list) -> tuple:

    feature_names = ["text_length", "av_word_length", "lexical_density"]
    features = []
    for revision in rev_list:
        features.append([len(revision), sum([len(token) for token in revision.split()])/len(revision.split()), compute_lexical_density(revision)])

    return features, feature_names


def prepare_linear_classifier(gold_standard: list, mode: str, add_ling: bool) -> list:
    # prepare the data for classification with linear classifiers
    # i.e. extract insertions and targets from gold file

    train_test = []
    for i, gold in enumerate(gold_standard):
        with open(gold, "r") as gs:
            samples = json.load(gs)

        revisions, targets, ids = [], [], []

        for k,sample in samples.items():
            revisions.append(re.findall(r"<(.*)>", sample["Revised"])[0])
            targets.append(sample["Majority Vote"])
            ids.append(k)

        if mode == "pos" or mode == "both":
            if i == 0:
                vectorizer = TfidfVectorizer(ngram_range=(2,3), token_pattern=r"[^ ]+", min_df=2)
                pos_features = vectorizer.fit_transform(pos_tagger(revisions))
            else:
                pos_features = vectorizer.transform(pos_tagger(revisions))
        if mode == "lex" or mode == "both":
            if i == 0:
                vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
                lex_features = vectorizer.fit_transform(revisions)
            else:
                lex_features = vectorizer.transform(revisions)

        feature_names = vectorizer.get_feature_names_out()
        if mode == "both":
            vector = hstack([pos_features, lex_features])
        elif mode == "pos":
            vector = pos_features
        elif mode == "lex":
            vector = lex_features

        if add_ling:
            ling_features, ling_feature_names = get_metrics(revisions)
            vector = hstack([vector, ling_features], format="csr")
            feature_names.tolist().append(ling_feature_names)

        train_test.append([vector, targets, ids])

    return train_test, feature_names