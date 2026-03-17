import os
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from datetime import date
from collections import Counter


def generate_plots(joint_df: pd.DataFrame, relevant_column: str, users: list, plot_name: str, study_path: str) -> None:

    def generate_bar_chart(df: pd.DataFrame):
        # Stacked bar chart
        # summary_df.plot(kind='bar', stacked=True, color=['#16c5d8', '#ffcc99', '#b23b3b'])
        df.plot(kind='bar', stacked=True, color=['#16c5d8', '#b23b3b'])
        plt.ylabel('Count')
        plt.legend(title=None)
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(f"{study_path}plots/bar_chart_{plot_name}_{str(date.today())}.png")

    def generate_pie_chart(df: pd.DataFrame):

        # Pie chart (aggregate)
        totals = df.sum()
        plt.pie(
            totals, 
            labels=totals.index, 
            autopct='%.1f%%', 
            # colors=['#16c5d8', '#ffcc99', '#b23b3b']
            colors=['#16c5d8', '#b23b3b']
        )
        plt.savefig(f"{study_path}plots/pie_chart_{plot_name}_{str(date.today())}.png")

    # Prepare a DataFrame to store results
    summary = {}

    for user in users:
        col = joint_df[relevant_column + f"_{user}"]
        counts = {
            "Yes": (col == 1).sum(),
            "No": (col == 0).sum()
        }
        summary[f"User_{user}"] = counts

    # Convert to DataFrame for plotting
    summary_df = pd.DataFrame(summary).T

    generate_pie_chart(summary_df)
    generate_bar_chart(summary_df)


def plot_entropy(entropies: list, study_path: str, out: str) -> None:

    # Plot the entropies
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(entropies)), entropies, color='skyblue')
    plt.xlabel('Item Index')
    plt.ylabel('Entropy (bits)')
    plt.title('Label Entropy per Annotated Item')
    plt.xticks(range(len(entropies)))
    # plt.ylim(0, 1.5)  # Adjust based on expected max entropy (log2 of number of labels)

    plt.tight_layout()
    plt.savefig(f"{study_path}plots/label_entropies_{out}_{str(date.today())}.png")


def plot_label_disagreement(disagreement_proportions: dict, study_path: str, out: str) -> None:

    # Plotting
    labels = list(disagreement_proportions.keys())
    values = [val/100 for val in disagreement_proportions.values()]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color='indigo')
    plt.ylabel('Item Ratio')
    plt.title('Distribution of Label Disagreement Levels')
    # plt.ylim(0, 0.35)

    # Add percentage labels to each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, round(yval, 3), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{study_path}plots/label_disagreement_{out}_{str(date.today())}.png")


def plot_label_distribution(labels: list, out: str, topic: str="") -> None:

    uniq_labels = sorted(list(set(labels)))
    labels_counts = [labels.count(label)/len(labels) for label in uniq_labels]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(uniq_labels, labels_counts, color='indigo')
    plt.ylabel('Label Ratio')
    plt.title('Distribution of Labels')
    # plt.ylim(0, 0.6)

    # Add percentage labels to each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, round(yval, 3), ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(out, f"{topic}_label_distribution_{str(date.today())}.png") if topic else os.path.join(out, f"label_distribution_{str(date.today())}.png")
    plt.savefig(save_path)


def plot_sample_annotations(samples_annotation: dict, study_path: str, out: str) -> None:

    color_dict = {"Yes": "orange", "No": "blue"}
    i = 0
    for key,annotation in samples_annotation.items():

        counts_implicit = Counter(annotation["label"])

        # Layout
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # bar chart and text

        # Bar chart on the left
        ax0 = plt.subplot(gs[0])
        ax0.bar(counts_implicit.keys(), counts_implicit.values(), color=[color_dict[label] for label in counts_implicit.keys()])
        ax0.set_xlabel("Response")
        ax0.set_ylabel("Count")
        ax0.set_title(f"Sample Original: {key}\nSample Revised: {annotation['sentence_2']}")

        comments_not_implicit = "\n\n".join([comment for comment in annotation["comment_not_implicit"] if comment])
        comments_implicit = "\n\n".join([comment for comment in annotation["comment_implicit"] if comment])
        # Text on the right
        ax1 = plt.subplot(gs[1])
        ax1.axis("off")  # hide axes
        text = (f"Comment FOR implicit meaning:\n\n{comments_implicit}\n\nComment AGAINST implicit meaning:\n\n{comments_not_implicit}")
        ax1.text(0, 1, text, va='top', ha='left', fontsize=11, wrap=True)

        plt.tight_layout()
        plt.savefig(f"{study_path}plots/samples/sample_{i}_{out}_{str(date.today())}.png")

        i += 1

    
def plot_linguistic_analysis(labels: list, values: Union[list, dict], ylabel: str, save_path: str, out: str, multibar_labels: list=[]):
    
    label_dict = {"No": "Implicit", "Yes": "New Information", "Tie": "Unclear"} 
    labels = [label_dict[label] for label in labels]
    if isinstance(values, list):
        plt.figure(figsize=(8, 4))
        print(values)
        bars = plt.bar(labels, values, color='mediumseagreen')
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across all labels.")
    else:
        values = np.array(list(values.values()))  # Shape: (num_categories, num_bars_per_group)

        n_bars = values.shape[1]
        bar_width = 0.2
        group_spacing = 0.3  # Space between groups

        # Recalculate x positions to include spacing between groups
        x = np.arange(len(labels)) * (n_bars * bar_width + group_spacing)
        # Create the plot
        fig, ax = plt.subplots()
        for i in range(n_bars):
            ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=f"{multibar_labels[i]}")

        # Customizing axes and layout
        ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percent points")
        ax.set_title("POS Distribution per Label")
        ax.legend()

    # Add percentage labels to each bar
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{save_path}/{out}_{str(date.today())}.png")


def plot_different_data_distributions(labels_data_1: list, labels_data_2: list, labels: list, multibar_labels: list, save_path: str):

    data_1 = [labels_data_1.count(label)/len(labels_data_1) for label in multibar_labels]
    data_2 = [labels_data_2.count(label)/len(labels_data_2) for label in multibar_labels]
    values = np.array([data_1, data_2])  # Shape: (num_categories, num_bars_per_group)

    n_bars = values.shape[1]
    bar_width = 0.2
    group_spacing = 0.3  # Space between groups

    # Recalculate x positions to include spacing between groups
    x = np.arange(len(labels)) * (n_bars * bar_width + group_spacing)
    # Create the plot
    fig, ax = plt.subplots()
    for i in range(n_bars):
        ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=f"{multibar_labels[i]}")

    # Customizing axes and layout
    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage points")
    ax.set_title("Label Distribution per Dataset")
    ax.legend()

    # Add percentage labels to each bar
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{save_path}_{str(date.today())}.png")

# ['No', 'No', 'No', 'Tie', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Tie', 'Tie', 'Yes', 'No', 'No', 'No', 'Tie', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'Tie', 'No', 'Tie', 'Yes', 'No', 'Tie', 'Tie', 'Tie', 'Yes', 'Tie', 'Tie', 'Tie', 'Tie', 'Tie', 'Yes', 'No', 'No', 'Yes', 'No', 'Tie', 'No', 'No', 'Yes', 'Tie', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Tie', 'No', 'No', 'Tie', 'Tie', 'Tie', 'No', 'Tie', 'No', 'Tie', 'Tie']
# ['Yes', 'Yes', 'No', 'Yes', 'Tie', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Tie', 'Tie', 'Yes', 'Tie', 'No']