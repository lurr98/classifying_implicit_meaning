import json, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('..')
from prepare_data import read_json, write_json


def get_rst_relation(rst_dict: dict, label_ids: list):

    label_relations = []
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


def check_overlap(rst_dict: dict, fn_dict: dict, label_ids: list):

    label_overlaps = []
    for i, label_id_list in enumerate(label_ids):
        label_overlaps.append({})
        for idx in label_id_list:
            if "relevant_segment" in rst_dict[idx]:
                if "relevant_frame" in fn_dict[idx]:
                    overlap = f"[{rst_dict[idx]['relevant_segment']['relname']}] – ({fn_dict[idx]['relevant_frame']['element']})"
                    if overlap in label_overlaps[i]:
                        label_overlaps[i][overlap][0] += 1
                        label_overlaps[i][overlap][1].append(idx)
                    else:
                        label_overlaps[i][overlap] = [1, [idx]]

    return label_overlaps


def plot_relations(relation_dicts: list, title: str, output_name: str, labels: list, percent: list | bool):

    exclude = {"span", "same-unit"}

    # --- collect all relations (rows) ---
    all_rels = sorted(set().union(*map(dict.keys, relation_dicts)) - exclude)

    # --- build matrix (relations x labels) ---
    if percent:
        # percent is a list of denominators (one per column)
        matrix = np.array([
            [ (relation_dicts[i].get(rel, [0])[0] / percent[i]) * 100
              for i in range(len(relation_dicts)) ]
            for rel in all_rels
        ], dtype=float)
        y_label = "Percentage (%)"
    else:
        matrix = np.array([
            [ d.get(rel, [0])[0] for d in relation_dicts ]
            for rel in all_rels
        ], dtype=float)
        y_label = "Count"

    df = pd.DataFrame(matrix, index=all_rels, columns=labels)

    # --- order by total descending ---
    df["Total"] = df.sum(axis=1)
    df = df.sort_values("Total", ascending=False)
    df_counts = df.drop(columns=["Total"])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_counts.index))
    bottom = np.zeros(len(df_counts), dtype=float)

    # store bottoms per layer for annotation
    bottoms_per_col = {}

    for col in df_counts.columns:
        heights = df_counts[col].values
        bars = ax.bar(df_counts.index, heights, bottom=bottom, label=col)
        bottoms_per_col[col] = bottom.copy()
        bottom = bottom + heights  # update for next stack

        # annotate this layer (inside the segment)
        if percent:
            for bar, btm, h in zip(bars, bottoms_per_col[col], heights):
                if h >= 1.0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        btm + h/2,
                        f"{round(h, 1)}%",
                        ha="center", va="center"
                    )

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_counts.index, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_name, dpi=200)
    plt.close(fig)


    # ---------- OPTIONAL: stacked percentages ----------
    # Uncomment to plot relative (%) composition per relation
    # pct = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    # plt.figure(figsize=(12, 6))
    # bottom = np.zeros(len(pct))
    # for col in pct.columns:
    #     plt.bar(pct.index, pct[col].values, bottom=bottom, label=col)
    #     bottom += pct[col].values
    # plt.xticks(rotation=45, ha="right")
    # plt.ylabel("Percentage (%)")
    # plt.title("RST relation distribution by label (stacked %)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('-a', '--analysis', action='store_true')
    parser.add_argument('-p', '--plotting', action='store_true')
    parser.add_argument('arguments', nargs='+', help='arguments needed for analysis or plotting (analysis: file containing parsed samples, gold standard, output file; plotting: file containg relation counts, output path)')
    parser.add_argument('-l', '--labels', nargs='?', help='list the labels in the data separated by commas')
    parser.add_argument('-perc', '--percent', action='store_true')


    args = parser.parse_args()

    labels = args.labels if args.labels else ["Implicit", "New Information", "Unclear"]

    if args.analysis:
        gold = read_json(args.arguments[1])

        label_list = []
        for label in sorted(labels):
            label_ids = []
            for idx, values in gold.items():
                if values["Majority Vote"] == label:
                    label_ids.append(idx)
            label_list.append(label_ids)

        if len(args.arguments[0].split(",")) == 2:
            framework1, framework2 = read_json(args.arguments[0].split(",")[0]), read_json(args.arguments[0].split(",")[1])

            output = check_overlap(framework1, framework2, label_list)
        else:
            framework = read_json(args.arguments[0])
            
            if "rst" in args.arguments[0]:
                output = get_rst_relation(framework, label_list)
            else:
                output = get_framenet_element(framework, label_list)

        write_json(output, args.arguments[2])

    elif args.plotting:
        relations = read_json(args.arguments[0])

        if "overlap" in args.arguments[0]:
            title = "[RST] and (Framenet) overlap counts by label"
        elif "rst" in args.arguments[0]:
            title = "RST relation counts by label"
        else:
            title = "Framenet element counts by label"

        if args.percent:
            percent = [104, 44, 68]
            plot_relations(relations, title, args.arguments[1], labels, percent)
        else:
            plot_relations(relations, title, args.arguments[1], labels, args.percent)