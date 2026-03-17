import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import re, spacy, argparse, json
from collections import Counter, defaultdict
from core.utils import read_json, write_json
from plot_script import plot_linguistic_analysis, plot_label_distribution

selected_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET"]
selection_dict = {"POS_distr": selected_POS}

nlp = spacy.load("en_core_web_sm")

def compute_pos_distribution(revision: str):
    doc = nlp(revision)
    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    
    pos_counts = Counter(token.pos_ for token in tokens)
    total = sum(pos_counts.values())

    pos_distribution = {pos: round(count / total, 3) for pos, count in pos_counts.items()}
    return pos_distribution

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


def get_revision_label(gold_standard: dict, mj: bool=False) -> list:

    rev_list = []
    for id, sample in gold_standard.items():
        rev = re.findall(r"<(.*)>", sample["Revised"])
        if mj:
            rev_list.append((id, rev[0], sample["Majority Vote"]))
        else:
            rev_list.append((id, rev[0], sample["MACE Label"]))

    return rev_list


def get_metrics(rev_list: list, metric_dict: str) -> None:

    save_metrics = {"IDs": [], "targets": [], "text_length": [], "av_word_length": [], "lexical_density": [], "POS_distr": []}

    for (id, revision, target) in rev_list:

        save_metrics["IDs"].append(id)
        save_metrics["targets"].append(target)
        # text length
        save_metrics["text_length"].append(len(revision))
        # average word length
        save_metrics["av_word_length"].append(sum([len(token) for token in revision.split()])/len(revision.split()))
        # lexical density
        save_metrics["lexical_density"].append(compute_lexical_density(revision))
        # POS distribution
        save_metrics["POS_distr"].append(compute_pos_distribution(revision))

    write_json(save_metrics, metric_dict)


def find_tendencies(metric_dict: dict, save_path: str, out: str, selected: dict, plot_dict: dict={"text_length": "Average Text Length of Addition", "av_word_length": "Average Word Length within Addition", "lexical_density": "Average Lexical Density of Addition", "POS_distr": "Average Distribution of POS tags within Addition"}) -> None:

    labels = list(set(metric_dict["targets"]))
    labels.sort()

    metric_target_split = {label: {"text_length": [], "av_word_length": [], "lexical_density": [], "POS_distr": []} for label in labels}
    for i, target in enumerate(metric_dict["targets"]):
        for key in list(metric_target_split[target].keys()):
            metric_target_split[target][key].append(metric_dict[key][i])

    for metric in list(metric_target_split[labels[0]].keys()):
        try:
            av_metrics = [sum(metric_target_split[label][metric])/len(metric_target_split[label][metric]) for label in labels]
            multibar = []
        except TypeError:
            av_metrics = average_categories(selected[metric], metric, metric_target_split)
            multibar = selected[metric]

        print(av_metrics)
        print(multibar)
        # plot_linguistic_analysis(labels, av_metrics, plot_dict[metric], save_path, out + metric, multibar_labels=multibar)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('-gl', '--gold_labels', type=str, help='name of the file containing the samples along with the gold labels')
    parser.add_argument('-md', '--metric_dict', type=str, help='name of the folder containing the linguistic analysis')
    parser.add_argument('-dd', '--data_distribution', action='store_true')
    parser.add_argument('-t', '--by_topic', action='store_true')
    parser.add_argument('-s', '--by_split', action='store_true')
    parser.add_argument('-mj', '--majority_vote', action='store_true')
    # parser.add_argument('output', type=str, help='name of the output file (in case of -gl) or path to output folder (in case of -md)')

    args = parser.parse_args()

    output = "metric_dict.json" 
    base_dir = "/anvme/workspace/v106be21-arr_workspace_december/"

    if args.gold_labels:

        labels = read_json(args.gold_labels)

        if args.by_topic:
            id2topic = read_json(os.path.join(base_dir, "implicit_data", "id2topic.json"))

            for topic in set(id2topic.values()):
                topic_labels = {idx: labels[idx] for idx, top in id2topic.items() if top == topic}
                topic_revisions = get_revision_label(topic_labels, args.majority_vote)
                get_metrics(topic_revisions, os.path.join(base_dir, "analysis", "metric_dicts", f"{topic}_{output}"))

                if args.data_distribution:
                    # TODO fix output name also in plot_script
                    plot_label_distribution([rev_tuple[2] for rev_tuple in topic_revisions], os.path.join(base_dir, "analysis", "plots", "label_distribution"), topic)
        elif args.by_split:
            data_splits = read_json(os.path.join(base_dir, "implicit_data", "splits.json"))

            for split in data_splits.keys():
                split_labels = {idx: labels[idx] for idx in data_splits[split]}
                split_revisions = get_revision_label(split_labels, args.majority_vote)
                get_metrics(split_revisions, os.path.join(base_dir, "analysis", "metric_dicts", f"{split}_{output}"))

                if args.data_distribution:
                    # TODO fix output name also in plot_script
                    plot_label_distribution([rev_tuple[2] for rev_tuple in split_revisions], os.path.join(base_dir, "analysis", "plots", "label_distribution"), split)

        else:
            revisions = get_revision_label(labels, args.majority_vote)
            get_metrics(revisions, os.path.join(base_dir, "analysis", "metric_dicts", output))

            if args.data_distribution:
                # TODO fix output name also in plot_script
                plot_label_distribution([rev_tuple[2] for rev_tuple in revisions], os.path.join(base_dir, "analysis", "plots", "label_distribution"))
        
    elif args.metric_dict:
        for md in os.listdir(args.metric_dict):
            metrics = read_json(os.path.join(args.metric_dict, md))

            save_path = os.path.join(base_dir, "analysis", "plots", "ling_analysis")
            find_tendencies(metrics, save_path, md.split("metric_dict")[0], selection_dict)