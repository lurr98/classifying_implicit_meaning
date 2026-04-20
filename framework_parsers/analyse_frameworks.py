import json, sys, argparse, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm   # colour map helper
from matplotlib.patches import Patch
from core.utils import read_json, write_json, append_json

label_dict = {"No": "Implicit", "Yes": "New Information"}
topics = ["arts", "business", "cars", "computers", "education", "family", "food", "garden", "health", "hobbies", "pets", "philosophy", "relationships", "sports", "style", "travel", "work", "youth"]
setting_map = {"few2": "Few-shot (2)", "few4": "Few-shot (4)", "few8": "Few-shot (8)", "few16": "Few-shot (16)", "few32": "Few-shot (32)", "few64": "Few-shot (64)", "few128": "Few-shot (128)", "zero": "Zero-shot/Off-the-shelf", "best_ft": "Fine-Tuned", "ft": "Fine-Tuned", "ft_depr": "Fine-Tuned (depr)"}
model_map = {"gpt": "GPT-5.2", "deepseek": "DeepSeek", "mistral": "Mistral", "llama": "Llama", "qwen": "Qwen", "bart": "BART-large", "roberta": "RoBERTa-large", "deberta": "DeBERTa-large"}

plt.rcParams['hatch.linewidth'] = 0.2


def get_rst_relation(rst_dict: dict, label_ids: list):

    label_relations = []
    print(label_ids)
    for i, label_id_list in enumerate(label_ids):
        label_relations.append({})
        for idx in label_id_list:
            if "relevant_segment" in rst_dict[idx]:
                if rst_dict[idx]["relevant_segment"]["relname"] in label_relations[i]:
                    label_relations[i][rst_dict[idx]["relevant_segment"]["relname"]][0] += 1
                    label_relations[i][rst_dict[idx]["relevant_segment"]["relname"]][1].append(idx)
                else:
                    label_relations[i][rst_dict[idx]["relevant_segment"]["relname"]] = [1, [idx]]

    return label_relations


def get_framenet_element(fn_dict: dict, label_ids: list):

    label_elements = []
    for i, label_id_list in enumerate(label_ids):
        label_elements.append({})
        for idx in label_id_list:
            if "relevant_frame" in fn_dict[idx]:
                if fn_dict[idx]["relevant_frame"]["element"] in label_elements[i]:
                    label_elements[i][fn_dict[idx]["relevant_frame"]["element"]][0] += 1
                    label_elements[i][fn_dict[idx]["relevant_frame"]["element"]][1].append(idx)
                else:
                    label_elements[i][fn_dict[idx]["relevant_frame"]["element"]] = [1, [idx]]

    return label_elements


def get_constituent_type(const_dict: dict, label_ids: list):

    constituents = []
    for i, label_id_list in enumerate(label_ids):
        constituents.append({})
        for idx in label_id_list:
            if "rev_phrase" in const_dict[idx]:
                if const_dict[idx]["rev_phrase"] in constituents[i]:
                    constituents[i][const_dict[idx]["rev_phrase"]][0] += 1
                    constituents[i][const_dict[idx]["rev_phrase"]][1].append(idx)
                else:
                    constituents[i][const_dict[idx]["rev_phrase"]] = [1, [idx]]
    return constituents



def check_overlap(first_dict: dict, second_dict: dict, label_ids: list, third_dict: dict=None):

    def add_overlap(overlap_dict: dict, overlap: str, idx: str):
        if overlap in overlap_dict:
            overlap_dict[overlap][0] += 1
            overlap_dict[overlap][1].append(idx)
        else:
            overlap_dict[overlap] = [1, [idx]]
        return overlap_dict

    label_overlaps = []
    for i, label_id_list in enumerate(label_ids):
        label_overlaps.append({})
        for idx in label_id_list:
            if third_dict:
                if "relevant_segment" in first_dict[idx]:
                    if "relevant_frame" in second_dict[idx]:
                        overlap = f"[{first_dict[idx]['relevant_segment']['relname']}] – ({second_dict[idx]['relevant_frame']['element']}) – <{third_dict[idx]['rev_phrase']}>"
                        label_overlaps[i] = add_overlap(label_overlaps[i], overlap, idx)
            elif "rev_phrase" in first_dict[idx]:
                if "relevant_segment" in second_dict[idx]:
                    overlap = f"[{first_dict[idx]['rev_phrase']}] – ({second_dict[idx]['relevant_segment']['relname']})"
                    label_overlaps[i] = add_overlap(label_overlaps[i], overlap, idx)
                else:
                    if "relevant_frame" in second_dict[idx]:
                        overlap = f"[{first_dict[idx]['rev_phrase']}] – ({second_dict[idx]['relevant_frame']['element']})"
                        label_overlaps[i] = add_overlap(label_overlaps[i], overlap, idx)
            else:
                if "relevant_segment" in first_dict[idx]:
                    if "relevant_frame" in second_dict[idx]:
                        overlap = f"[{first_dict[idx]['relevant_segment']['relname']}] – ({second_dict[idx]['relevant_frame']['element']})"
                        label_overlaps[i] = add_overlap(label_overlaps[i], overlap, idx)

    return label_overlaps


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm   # colour map helper


def plot_relations(
    relation_dicts: list,
    title: str,
    output_name: str,
    labels: list,
    percent: bool,
):
    """
    Plot a **grouped** bar chart (bars for each label sit next to each other)
    showing either raw counts or percentages for the supplied relation dictionaries.
    """

    # ------------------------------------------------------------------ #
    # 1️⃣  Configuration
    # ------------------------------------------------------------------ #
    exclude = {"span", "same-unit"}                     # relations we never show
    # nice label names – assumes a global ``label_dict`` exists
    labels = [label_dict.get(l, l) for l in labels]

    # ------------------------------------------------------------------ #
    # 2️⃣  Percent conversion (if requested)
    # ------------------------------------------------------------------ #
    if percent:
        # compute column totals once; keep the original boolean flag untouched
        denom_no = sum(v[0] for v in relation_dicts[0].values())
        denom_yes = sum(v[0] for v in relation_dicts[1].values())
        denominators = [denom_no, denom_yes]
    else:
        denominators = None

    # ------------------------------------------------------------------ #
    # 3️⃣  Gather the set of relations (rows) we will plot
    # ------------------------------------------------------------------ #
    all_rels = sorted(set().union(*map(dict.keys, relation_dicts)) - exclude)

    # ------------------------------------------------------------------ #
    # 4️⃣  Build the numeric matrix (relations × labels)
    # ------------------------------------------------------------------ #
    if percent:
        matrix = np.array(
            [
                [
                    (relation_dicts[i].get(rel, [0])[0] / denominators[i]) * 100
                    if denominators[i] > 0 else 0.0
                    for i in range(len(relation_dicts))
                ]
                for rel in all_rels
            ],
            dtype=float,
        )
        y_label = "Percentage (%)"
    else:
        matrix = np.array(
            [[d.get(rel, [0])[0] for d in relation_dicts] for rel in all_rels],
            dtype=float,
        )
        y_label = "Count"

    # Turn the matrix into a DataFrame for convenient sorting / slicing
    df = pd.DataFrame(matrix, index=all_rels, columns=labels)

    # Keep only the most frequent relations (top‑8) – same behaviour as before
    df["Total"] = df.sum(axis=1)
    df = df.sort_values("Total", ascending=False).drop(columns="Total").head(5)

    # ------------------------------------------------------------------ #
    # 5️⃣  Plot – grouped bars
    # ------------------------------------------------------------------ #
    n_groups = len(df)                 # number of distinct relations
    n_labels = len(labels)             # number of bars per group

    # Width of a single bar; the whole group will occupy ~0.8 of the x‑tick space
    bar_width = 0.8 / n_labels
    # Positions of the group centres on the x‑axis
    indices = np.arange(n_groups)

    # Choose colours from the tab20 palette (first colour for label 0, second for label 1, …)
    cmap = cm.get_cmap("tab20")
    colours = [cmap(i) for i in range(n_labels)]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot each label side‑by‑side
    for i, (lbl, col) in enumerate(zip(labels, colours)):
        # Shift each bar within its group
        offset = (i - n_labels / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            indices + offset,
            df[lbl].values,
            width=bar_width,
            label=lbl,
            color=col,
            edgecolor="black",
        )

        for bar in bars:
            height = bar.get_height()
            # Show one decimal for percentages, none for raw counts
            txt = f"{height:.1f}" if percent else f"{int(height)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,   # centre of the bar (x)
                height / 2,                         # vertical centre of the bar
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )  # ← add label
    # ------------------------------------------------------------------ #
    # 6️⃣  Finishing touches
    # ------------------------------------------------------------------ #
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # X‑tick labels are the relation names
    ax.set_xticks(indices)
    ax.set_xticklabels(df.index, rotation=45, ha="right")

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_name, dpi=200)
    plt.close(fig)


def plot_relations_by_setting(
    data_by_setting: dict,
    title: str,
    output_name: str,
    labels: list,
    percent: bool
):
    exclude = {"span", "same-unit"}
    labels_pretty = [label_dict[lbl] if lbl in label_dict else lbl for lbl in labels]

    # output base + extension
    if "." in output_name:
        base, ext = output_name.rsplit(".", 1)
        ext = "." + ext
    else:
        base, ext = output_name, ".png"

    # tab20 gives 20 colors = 10 pairs
    cmap = plt.get_cmap("tab20")
    max_pairs = cmap.N // 2  # 10

    for setting, (models, label_dicts) in data_by_setting.items():
        if not setting in setting_map:
            continue

        n_models = len(models)
        n_labels = len(label_dicts)

        if n_labels != 2:
            raise ValueError("This paired-color version expects exactly 2 labels (two stacked segments).")

        if n_models > max_pairs:
            raise ValueError(f"tab20 supports {max_pairs} paired model colors (max {max_pairs} models), got {n_models}.")

        # --- per-model paired colors: (dark, light) ---
        model_pair = {}
        for i, m in enumerate(models):
            dark = cmap(2 * i)
            light = cmap(2 * i + 1)
            model_pair[m] = (dark, light)

        # --- collect relations ---
        all_rels = sorted(set().union(*[d.keys() for d in label_dicts]) - exclude)

        # --- counts: (rels x models x labels) ---
        counts = np.zeros((len(all_rels), n_models, n_labels), dtype=float)
        for r_idx, rel in enumerate(all_rels):
            for l_idx in range(n_labels):
                per_model = label_dicts[l_idx].get(rel, [0] * n_models)
                if len(per_model) < n_models:
                    per_model = list(per_model) + [0] * (n_models - len(per_model))
                for m_idx in range(n_models):
                    counts[r_idx, m_idx, l_idx] = per_model[m_idx]

        # --- percent or counts ---
        if percent:
            denom = counts.sum(axis=0)  # (models x labels)
            values = np.zeros_like(counts)
            for m_idx in range(n_models):
                for l_idx in range(n_labels):
                    d = denom[m_idx, l_idx]
                    values[:, m_idx, l_idx] = (counts[:, m_idx, l_idx] / d) * 100 if d > 0 else 0.0
            y_label = "Percentage (%)"
        else:
            values = counts
            y_label = "Count"

        # --- top 10 by raw total ---
        t = 4 if "constituency" in title.lower() else 8
        total_raw = counts.sum(axis=(1, 2))
        top_idx = np.argsort(-total_raw)[:t]
        rels_top = [all_rels[i] for i in top_idx]
        values_top = values[top_idx, :, :]  # (top_rels x models x labels)

        # --- plot grouped + stacked ---
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(rels_top))
        group_width = 0.8
        bar_width = group_width / max(n_models, 1)
        offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

        for m_idx, model in enumerate(models):
            x_m = x + offsets[m_idx]
            bottom = np.zeros(len(rels_top), dtype=float)

            dark, light = model_pair[model]
            colors_for_labels = [dark, light]

            for l_idx in range(n_labels):
                heights = values_top[:, m_idx, l_idx]
                bars = ax.bar(
                    x_m,
                    heights,
                    width=bar_width,
                    bottom=bottom,
                    hatch="++" if model in ["bart", "roberta", "deberta"] else None,  # pattern for NLI models
                    color=colors_for_labels[l_idx],
                    edgecolor="black",
                    linewidth=0.6
                )

                # annotate percentages/counts
                # if percent:
                for bar, btm, h in zip(bars, bottom, heights):
                    if h >= 0.5:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            btm + h / 2,
                            # round to 0 if last digit is 0, else 1 decimal place
                            round(h, 1) if round(h, 1) != int(h) else int(h),
                            ha="center",
                            va="center",
                            fontsize=5
                        )

                bottom += heights

        ax.set_xticks(x)
        ax.set_xticklabels(rels_top, rotation=45, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} — {setting_map[setting]}")

        # --- legends ---
        # Model legend: use the *darker* shade as the model identifier
        model_handles = [
            Patch(facecolor=model_pair[m][0], edgecolor="black", label=model_map[m])
            for m in models
        ]

        # Label legend: show dark vs light using the first model's pair (explains the scheme)
        example_dark, example_light = model_pair[models[0]]
        label_handles = [
            Patch(facecolor=example_dark, edgecolor="black", label=labels_pretty[0]),
            Patch(facecolor=example_light, edgecolor="black", label=labels_pretty[1]),
        ]

        leg1 = ax.legend(handles=label_handles, title="Label (shade)", loc="upper right")
        ax.add_artist(leg1)
        ax.legend(handles=model_handles, title="Model", loc="upper right", bbox_to_anchor=(1, 0.85))

        fig.tight_layout()
        out_path = f"{base}_{setting}{ext}" if percent else f"{base}_{setting}_counts{ext}"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def synthesize_relations(aggregated_rel_dict_file: str):

    aggregated_rel_dict = read_json(aggregated_rel_dict_file)

    setting_syn = {}
    for setting, model_rels in aggregated_rel_dict.items():
        synthesized, model_names = [{}, {}], []
        for model, label_relations in model_rels.items():
            model_names.append(model)
            for i, label_relation in enumerate(label_relations):
                for rel, values in label_relation.items():
                    if rel in synthesized[i]:
                        synthesized[i][rel].append(values[0])
                    else:
                        zero_fill = [0] * len(model_names)
                        zero_fill[-1] = values[0]
                        synthesized[i][rel] = zero_fill
                        # synthesized[i][rel] = [values[0]]

                # add relations missing in this model with count 0
                for rel in synthesized[i].keys():
                    if rel not in label_relation:
                        synthesized[i][rel].append(0)
        setting_syn[setting] = (model_names, synthesized)

    return setting_syn  



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('-a', '--analysis', action='store_true')
    parser.add_argument('-p', '--plotting', action='store_true')
    parser.add_argument('arguments', nargs='+', help='arguments needed for analysis or plotting (analysis: file containing parsed samples, gold standard, output file; plotting: file containg relation counts, output path)')
    parser.add_argument('-l', '--labels', nargs='?', help='list the labels in the data separated by commas')
    parser.add_argument('-t', '--topic', nargs='?', help='the topic to analyze')
    parser.add_argument('-s', '--split', nargs='?', help='the split to analyze')
    parser.add_argument('-m', '--mistakes', nargs='?', help='the file containing the mistakes')
    parser.add_argument('-perc', '--percent', action='store_true')

    base_dir = "/home/vault/v106be/v106be21/classifying_implicit_meaning/framework_parsers"

    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else ["Implicit", "New Information"]

    if args.analysis:

        gold = read_json(args.arguments[1])

        if args.topic:
            id2topic = read_json("/anvme/workspace/v106be21-arr_workspace_december/implicit_data/id2topic.json")
            gold = {idx: sample for idx, sample in gold.items() if id2topic[idx] == args.topic} 

        if args.split:
            data_splits = read_json("/anvme/workspace/v106be21-arr_workspace_december/implicit_data/splits.json")
            gold = {idx: gold[idx] for idx in data_splits[args.split]} 

        label_list = []
        for label in sorted(labels):
            label_ids = []
            for idx, values in gold.items():
                if values["MACE Label"] == label:
                    label_ids.append(idx)
            label_list.append(label_ids)

        if len(args.arguments[0].split(",")) == 2:
            framework1, framework2 = read_json(args.arguments[0].split(",")[0]), read_json(args.arguments[0].split(",")[1])

            output = check_overlap(framework1, framework2, label_list)
        elif len(args.arguments[0].split(",")) == 3:
            framework1, framework2, framework3 = read_json(args.arguments[0].split(",")[0]), read_json(args.arguments[0].split(",")[1]), read_json(args.arguments[0].split(",")[2])

            output = check_overlap(framework1, framework2, label_list, framework3)
        else:
            framework = read_json(args.arguments[0])
            
            if "rst" in args.arguments[0]:
                output = get_rst_relation(framework, label_list)
            elif "framenet" in args.arguments[0]:
                output = get_framenet_element(framework, label_list)
            else:
                output = get_constituent_type(framework, label_list)
        
        out = os.path.join(base_dir, args.arguments[2])
        write_json(output, out)

    elif args.plotting:
        relations = read_json(args.arguments[0])

        if args.percent:
            cp = "percentage"
            id2label = read_json("/anvme/workspace/v106be21-arr_workspace_december/implicit_data/id2label.json")
            if args.topic:
                id2topic = read_json("/anvme/workspace/v106be21-arr_workspace_december/implicit_data/id2topic.json")
                topic_id2label = {label: [idx for idx in id2label[label] if id2topic[idx] == args.topic] for label in id2label.keys()}
                id2label = topic_id2label

            if args.split:
                data_splits = read_json("/anvme/workspace/v106be21-arr_workspace_december/implicit_data/splits.json")
                split_id2label = {label: [idx for idx in id2label[label] if idx in data_splits[args.split]] for label in id2label.keys()}
                id2label = split_id2label

            percent = [len(id2label["No"]), len(id2label["Yes"])]
            # plot_relations(relations, title, out, labels, percent)
        else:
            # plot_relations(relations, title, out, labels, args.percent)
            cp = "counts"
            percent = args.percent


        if "overlap" in args.arguments[0]:
            title = f"[RST] and (Framenet) overlap {cp} by label"
            parser = "overlap"
        elif "rst" in args.arguments[0]:
            title = f"RST relation {cp} by label"
            parser = "rst"
        elif "framenet" in args.arguments[0]:
            title = f"Framenet element {cp} by label"
            parser = "framenet"
        else:
            title = f"Constituency phrase {cp} by label"
            parser = "constituency"

        out = os.path.join(base_dir, "plots", args.arguments[1])

        if args.mistakes:
            if args.mistakes.startswith("all_model_mistakes_"):
                syn_dict = synthesize_relations(args.mistakes)
                # print(syn_dict["zero"])
                plot_relations_by_setting(
                    syn_dict,
                    title,
                    out,
                    labels,
                    percent
                )
            else:
                mistakes = read_json(args.mistakes)
                model = args.mistakes.split("_")[-1][:-5]
                for setting in mistakes.keys():
                    if "_" in setting and setting.split("_")[-1] in topics:
                        continue
                    new_relations = []
                    for i, label_analysis in enumerate(relations):
                        new_rel_analysis = {}
                        for rel, values in label_analysis.items():
                            new_idxs = [idx for idx in values[1] if idx in mistakes[setting]]
                            if len(new_idxs) > 0:
                                new_rel_analysis[rel] = [len(new_idxs), new_idxs]
                        new_relations.append(new_rel_analysis)
                    try:
                        append_json({setting: {model: new_relations}}, f"all_model_mistakes_{parser}.json")
                    except FileNotFoundError:
                        write_json({setting: {model: new_relations}}, f"all_model_mistakes_{parser}.json")
                    # plot_relations(new_relations, title, out[:-4] + f"_mistakes_{model}_{setting}.png", labels, percent)
        else:
            plot_relations(relations, title, out, labels, percent)
