import argparse, json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def plot_confusion_matrix(gold_labels: list, predictions: list, labels: list, plot_name: str) -> None:

    plot_label_dict = {"No": "Implicit", "Yes": "New", "Tie": "Tie", "UNK": "UNK"}

    gold_labels = [plot_label_dict[label] for label in gold_labels]
    predictions = [plot_label_dict[label] for label in predictions]
    labels = [plot_label_dict[label] for label in labels]
    cm = confusion_matrix(gold_labels, predictions, labels=sorted(labels))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(labels))
    disp.plot()

    plt.savefig(plot_name, bbox_inches='tight')


def plot_metric_curve(results: dict, desired_experiments: list, few_shots: list, noties: bool, plot_name: str, metric: str, data_portion: str):
    
    noties_str = "noties_" if noties else ""
    dp_str = f"_{data_portion}" if data_portion else ""

    plt.figure(figsize=(10, 6))

    for exp in desired_experiments:
        metrics = []

        metrics.append(results[f"{exp}_zero_{noties_str}1007{dp_str}"][metric])
        for few in few_shots:
            metrics.append(results[f"{exp}_few{few}_{noties_str}1007{dp_str}"][metric])
            # TODO: continue here

        # Sort by experiment names or accuracy if you prefer
        # experiments, accuracies = zip(*sorted(zip(experiments, accuracies)))

        plt.plot([0]+few_shots, metrics, marker='o', linestyle='-', label=exp)

    plt.title(f"{metric.title()} Curve")
    plt.xlabel("Number of Few-Shot Examples")
    plt.ylabel(metric.title())
    plt.ylim(0, 1)
    # plt.xticks([0] + few_shots)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{plot_name}_{noties_str}{metric}{dp_str}.png")


def plot_metric_curve_crossval(results: dict, desired_experiments: list, test_batch: str, noties: bool, plot_name: str, metric: str, data_portion: str):
    
    noties_str = "noties_" if noties else ""
    dp_str = f"_{data_portion}" if data_portion else ""

    plt.figure(figsize=(10, 6))

    for exp in desired_experiments:
        metrics = []

        metrics.append(results[f"{exp}_zero_{noties_str}{test_batch}{dp_str}"][metric])
        metrics.append(results[f"{exp}_few_{noties_str}{test_batch}{dp_str}"][metric])

        plt.plot(["0", f"108 (Test Batch: {test_batch})"], metrics, marker='o', linestyle='-', label=exp)

    plt.title(f"{metric.title()} Curve")
    plt.xlabel("Number of Few-Shot Examples")
    plt.ylabel(metric.title())
    plt.ylim(0, 1)
    plt.xticks([0, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{plot_name}_{noties_str}{metric}{test_batch}{dp_str}.png")


def plot_nli_metrics(results: dict, desired_experiments: list, metrics: str, plot_name: str, data_portion: str, secondprob: bool):

    # === 1) Overall metrics ===

    overall_data = []
    dp_str = f"_{data_portion}" if data_portion else ""
    for experiment in desired_experiments:
        exp_name = f"mergedstudy_{experiment}_1007_secondprob{dp_str}" if secondprob else f"mergedstudy_{experiment}_1007{dp_str}"
        values = [results[exp_name][metric] for metric in metrics]
        overall_data.append(values)

    overall_data = np.array(overall_data)

    x = np.arange(len(metrics))
    num_models = len(desired_experiments)
    width = 0.8 / num_models  # ensures groups don’t overflow

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(desired_experiments):
        offset = -0.4 + i*width + width/2
        plt.bar(x + offset, overall_data[i], width, label=experiment.title())

    plt.xticks(x + width / 2, metrics)
    plt.ylim(0, 1.2)
    plt.title("Overall Metrics Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    if secondprob:
        plt.savefig(f"{plot_name}_secondprob{dp_str}.png")
    else:
        plt.savefig(f"{plot_name}_{dp_str}.png")

    # === 2) Per-class metrics ===
    per_class_metrics = ["precision_class", "recall_class", "fscore_class"]
    if data_portion in ["nt", "cc"]:
        classes = ["no", "yes"]
    else:
        classes = ["no", "tie", "yes"]

    for metric in per_class_metrics:
        plt.figure(figsize=(10, 6))
        for i, experiment in enumerate(desired_experiments):
            exp_name = f"mergedstudy_{experiment}_1007_secondprob{dp_str}" if secondprob else f"mergedstudy_{experiment}_1007{dp_str}"
            values = [results[exp_name][metric][f"{metric.split('_')[0]}_{label}"] for label in classes]
            x = np.arange(len(classes))
            offset = -0.4 + i*width + width/2
            plt.bar(x + offset, values, width, label=experiment.title())

        plt.xticks(x + width / 2, [label.title() for label in classes])
        plt.ylim(0, 1.2)
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        if secondprob:
            plt.savefig(f"{plot_name}_{metric}_secondprob{dp_str}.png")
        else:
            plt.savefig(f"{plot_name}_{metric}{dp_str}.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('evaluation_dict', type=str, help='name of the file containing the evaluation')
    parser.add_argument('experiments', type=str, help='name of the experiment that was conducted, multiple separated by commas if more than one is requested')
    parser.add_argument('plot_name', type=str, help='name of the plot(s) to be generated, does not require doc type extension')
    parser.add_argument('-f', '--few_shots', nargs='?', help='requested few shot numbers separated by commas')
    parser.add_argument('-m', '--metric', nargs='?', help='metric in case another metric than accuracy is requested')
    parser.add_argument('-dp', '--data_portion', nargs='?', help='name of requested data portion')
    parser.add_argument('-tb', '--test_batch', nargs='?', help='name of requested data portion')
    parser.add_argument('-nt', '--no_ties', action='store_true')
    parser.add_argument('-sp', '--second_prob', action='store_true')

    # TODO: add addition e.g. cc
    args = parser.parse_args()

    with open(args.evaluation_dict, "r") as ev:
        evaluation = json.load(ev)

    if args.few_shots:
        metric = args.metric if args.metric else "accuracy"
        if "," in args.experiments:
            experiments = '_'.join(args.experiments.split(","))
            plot_metric_curve(evaluation, args.experiments.split(","), args.few_shots.split(","), args.no_ties, args.plot_name + f"_{experiments}", metric, args.data_portion)
        else:
            plot_metric_curve(evaluation, [args.experiments], args.few_shots.split(","), args.no_ties, args.plot_name + f"_{args.experiments}", metric, args.data_portion)

    elif args.test_batch:
        metric = args.metric if args.metric else "accuracy"
        if "," in args.experiments:
            experiments = '_'.join(args.experiments.split(","))
            plot_metric_curve_crossval(evaluation, args.experiments.split(","), args.test_batch, args.no_ties, args.plot_name + f"_{experiments}", metric, args.data_portion)
        else:
            plot_metric_curve_crossval(evaluation, [args.experiments], args.test_batch, args.no_ties, args.plot_name + f"_{args.experiments}", metric, args.data_portion)

    else:
        experiments = '_'.join(args.experiments.split(","))
        plot_nli_metrics(evaluation, args.experiments.split(","), args.metric.split(","), args.plot_name + f"_{experiments}", args.data_portion, args.second_prob)