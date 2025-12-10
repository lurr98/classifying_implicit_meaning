import argparse, json, re
import matplotlib.pyplot as plt
from plot_script import plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
from collections import defaultdict
from typing import Union


# TODO
def common_prefix(strings: list):
    if not strings:
        return ""
    s1 = min(strings)
    s2 = max(strings)
    for i, c in enumerate(s1):
        if i >= len(s2) or c != s2[i]:
            return s1[:i]
    return s1

def common_suffix(strings: list):
    reversed_strings = [s[::-1] for s in strings]
    reversed_prefix = common_prefix(reversed_strings)
    return reversed_prefix[::-1]


def get_prediction(pred: str, exp_name: str) -> str:

    if pred.split("\n")[0].strip() in ["No", "Yes", "Tie"]:
        return pred.split("\n")[0].strip()
    elif pred.split("\n")[0].strip() in ['"No"', '"Yes"', '"Tie"']:
        return pred.split("\n")[0].strip()[1:-1]
    else:
        if pred.split("\n")[0].strip() in ["Label: No", "Label: Yes", "Label: Tie"]:
            return pred.split("\n")[0].strip().split("Label: ")[1].strip()
        else:
            matches = []
            for label in ["No", "Tie", "Yes"]:
                matches.extend(re.findall(label, pred))
            if len(list(set(matches))) == 1:
                return matches[0]
            else:
                print(f"Unclear Label in Experiment {exp_name}: {pred}")
                return "UNK"


def evaluate_predictions_classification(gold_labels: dict, predictions: dict, evaluation_dict: dict, key: str, plot_name: str, second_pred: bool, mistakes: Union[bool, str]) -> dict:

    pred_list, gold_list, id_list = [], [], []
    for id_sample, sample in gold_labels.items():
        if id_sample in predictions:
            gold_list.append(sample["Majority Vote"])
            pred_list.append(get_prediction(predictions[id_sample], key))
            id_list.append(id_sample)

    if mistakes and key.endswith("nt"):
        handle_mistakes(mistakes, gold_list, pred_list, id_list, key)

    all_labels = sorted(list(set(gold_list)))
    
    print("Start evaluating.")
    accuracy = accuracy_score(gold_list, pred_list)
    fscores = {f"fscore_{setting}": f1_score(gold_list, pred_list, average=setting) for setting in ['micro', 'macro', 'weighted']}
    precision, recall, fscore, support = score(gold_list, pred_list, labels=all_labels)
    print("Ended evaluating.")
    precision_dict, recall_dict, fscore_dict = {"precision_class": {f"precision_{label.lower()}": precision[i] for i, label in enumerate(all_labels)}},\
                                                {"recall_class": {f"recall_{label.lower()}": recall[i] for i, label in enumerate(all_labels)}},\
                                                {"fscore_class": {f"fscore_{label.lower()}": fscore[i] for i, label in enumerate(all_labels)}}
                                                
    evaluation_dict[key] = {"accuracy": accuracy} | fscores | precision_dict | recall_dict | fscore_dict

    if plot_name:
        plot_confusion_matrix(gold_list, pred_list, sorted(list(set(gold_list) | set(pred_list))), plot_name)

    return evaluation_dict


def evaluate_predictions_nli(gold_labels: dict, predictions: dict, evaluation_dict: dict, key: str, plot_name: str, second_pred: str, mistakes: Union[bool, str]) -> dict:

    if len(list(predictions["predictions"].values())[0]) == 2:
        # TODO: not_entailment could also be "Tie"
        label_dict = {"not_entailment": "Yes", "entailment": "No"}
    else:
        label_dict = {"contradiction": "Yes", "neutral": "Tie", "entailment": "No"}
    pred_list, gold_list, id_list = [], [], []
    if second_pred:
        plot_name = plot_name[:-4] + f"_secondprob{second_pred}.png"
    for id_sample, sample in gold_labels.items():
        if id_sample in predictions["predictions"]: 
            gold_list.append(sample["Majority Vote"])
            preds = predictions["predictions"][id_sample]
            id_list.append(id_sample)
            if second_pred:
                if preds[0] > float(second_pred) or preds[2] > float(second_pred):
                    binary_preds = [preds[0], preds[2]]
                    pred_list.append(predictions["labels"][preds.index(max(binary_preds))])
                else:
                    pred_list.append(predictions["labels"][preds.index(max(preds))])
            else:
                pred_list.append(predictions["labels"][preds.index(max(preds))])

    pred_list = [label_dict[item] for item in pred_list]

    if mistakes:
        handle_mistakes(mistakes, gold_list, pred_list, id_list, key)

    all_labels = sorted(list(set(gold_list)))

    accuracy = accuracy_score(gold_list, pred_list)
    fscores = {f"fscore_{setting}": f1_score(gold_list, pred_list, average=setting) for setting in ['micro', 'macro', 'weighted']}
    precision, recall, fscore, support = score(gold_list, pred_list, labels=all_labels)
    
    precision_dict, recall_dict, fscore_dict = {"precision_class": {f"precision_{label.lower()}": precision[i] for i, label in enumerate(all_labels)}},\
                                                {"recall_class": {f"recall_{label.lower()}": recall[i] for i, label in enumerate(all_labels)}},\
                                                {"fscore_class": {f"fscore_{label.lower()}": fscore[i] for i, label in enumerate(all_labels)}}
                                                
    evaluation_dict[key] = {"accuracy": accuracy} | fscores | precision_dict | recall_dict | fscore_dict

    if plot_name:
        plot_confusion_matrix(gold_list, pred_list, sorted(list(set(gold_list) | set(pred_list))), plot_name)

    return evaluation_dict


def handle_mistakes(mistakes: str, gold_list: list, pred_list: list, id_list: list, key: str):

    with open(mistakes, "r") as m:
            mistake_dict = json.load(m)

    new_mistake_dict = store_mistakes(mistake_dict, gold_list, pred_list, id_list, key)

    with open(mistakes, "w") as nm:
        json.dump(new_mistake_dict, nm)


def store_mistakes(mistake_dict: dict, gold_list: list, pred_list: list, id_list: list, key: str):

    for i, gold_label in enumerate(gold_list):
        if gold_label != pred_list[i]:
            if key in mistake_dict:
                mistake_dict[key].append(id_list[i])
            else:
                mistake_dict[key] = [id_list[i]]

    print(f"Mistake dict: {key}")
    print(len(mistake_dict[key]))

    return mistake_dict



def get_clear_cases(gold_standard: dict) -> dict:

    clear_cases = {}
    for sample_id, sample in gold_standard.items():
        if sample["Label Distribution"][0] == sum(sample["Label Distribution"]) or sample["Label Distribution"][1] == sum(sample["Label Distribution"]):
            clear_cases[sample_id] = sample

    return clear_cases


def remove_ties(gold_standard: dict) -> dict:

    no_ties = {}
    i = 0
    for sample_id, sample in gold_standard.items():
        if sample["Majority Vote"] != "Tie":
            i += 1
            no_ties[sample_id] = sample

    return no_ties


def sub_sample(gold_standard: dict) -> dict:

    clear_cases = get_clear_cases(gold_standard)
    start = {}

    for sample_id, sample in gold_standard.items():
        # TODO: make this more general such that less frequent label is not necessarily "Yes" 
        if sample["Majority Vote"] == "Yes":
            start[sample_id] = sample

    final = clear_cases | start
    less_freq_samples = len(start.keys()) - (len(final.keys()) - len(start.keys()))
    if len(final.keys()) / 2 > len(start.keys()):
        i = len(final.keys()) - len(start.keys())
        for sample_id in list(final.keys()):
            if i > len(start.keys()):
                if final[sample_id]["Majority Vote"] == "No":
                    del final[sample_id]
                    i -= 1
    else:
        i = 0
        for sample_id, sample in gold_standard.items():
            if i < less_freq_samples:
                if sample["Majority Vote"] == "No" and not sample_id in final:
                    final[sample_id] = sample
                    i += 1

    return final


def average_metrics(results: dict) -> dict:

    # For storing sums
    sums = defaultdict(float)
    count = 0

    for entry in results.values():
        for metric, value in entry.items():
            sums[metric] += value
        count += 1

    # Compute averages
    averages = {metric: sums[metric] / count for metric in sums}
    return averages


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('targets', type=str, help='name of the file containing the target labels')
    parser.add_argument('predictions', type=str, help='name of the file(s) containing the predictions')
    parser.add_argument('evaluation_dict', type=str, help='name of the evaluation file')
    parser.add_argument('-m', '--mistake', nargs='?', help='name of the mistake file if needed')
    parser.add_argument('-sp', '--second_pred', nargs='?', help='threshold for second prediction')
    parser.add_argument('-nli', '--nli', action='store_true')
    parser.add_argument('-p', '--plot_name', action='store_true')
    parser.add_argument('-ss', '--sub_sample', action='store_true')
    parser.add_argument('-cc', '--clear_cases', action='store_true')
    parser.add_argument('-nt', '--no_ties', action='store_true')
    parser.add_argument('-cv', '--cross_val', action='store_true')

    args = parser.parse_args()

    print("this is running")
    print("Read targets.")
    with open(args.targets, "r") as trg:
        targets = json.load(trg)

    try:
        with open(args.evaluation_dict, "r") as ev:
            evaluation = json.load(ev)
    except FileNotFoundError:
        evaluation = {}

    save_eval_names = [[] for pred_file in args.predictions.split(",")]
    for i, file in enumerate(args.predictions.split(",")):
        with open(file, "r") as preds:
            predictions = json.load(preds)

        if args.second_pred:
            evaluation_name = "_".join(file.split("predictions_")[1:])[:-5] + f"_secondprob{args.second_pred}"
        else:
            evaluation_name = "_".join(file.split("predictions_")[1:])[:-5]

        if args.nli:
            evaluate = evaluate_predictions_nli
            nli, noties = "nli", ""
        else:
            evaluate = evaluate_predictions_classification
            nli = "classification"
            noties = "noties/" if "noties" in evaluation_name else "ties/"
    
        if "discarded" in args.targets:
            evaluation_name += "_discarded"
        elif "kept" in args.targets:
            evaluation_name += "_kept"
        save_eval_names[i].append(evaluation_name)
    
        if args.plot_name:
            new_evaluation = evaluate(targets, predictions, evaluation, evaluation_name, f"plots/{nli}/confusion_matrices/{noties}cm_" + evaluation_name + ".png", args.second_pred, args.mistake)
        else:
            new_evaluation = evaluate(targets, predictions, evaluation, evaluation_name, "", args.second_pred, args.mistake)
    
        if args.clear_cases:
            cc_targets = get_clear_cases(targets)
            cc_evaluation_name = evaluation_name + "_cc"
            save_eval_names[i].append(cc_evaluation_name)
    
            new_evaluation = evaluate(cc_targets, predictions, evaluation, cc_evaluation_name, f"plots/{nli}/confusion_matrices/{noties}cm_" + cc_evaluation_name + ".png" if args.plot_name else "", args.second_pred, args.mistake)
    
        if args.no_ties:
            nt_targets = remove_ties(targets)
            nt_evaluation_name = evaluation_name + "_nt"
            save_eval_names[i].append(nt_evaluation_name)
    
            new_evaluation = evaluate(nt_targets, predictions, evaluation, nt_evaluation_name, f"plots/{nli}/confusion_matrices/{noties}cm_" + nt_evaluation_name + ".png" if args.plot_name else "", args.second_pred, args.mistake)
    
        if args.sub_sample:
            ss_targets = sub_sample(targets)
            ss_evaluation_name = evaluation_name + "_ss"
            save_eval_names[i].append(ss_evaluation_name)
    
            new_evaluation = evaluate(ss_targets, predictions, evaluation, ss_evaluation_name, f"plots/{nli}/confusion_matrices/{noties}cm_" + ss_evaluation_name + ".png" if args.plot_name else "", args.second_pred, args.mistake)


    if args.cross_val:
        for j, name in enumerate(save_eval_names[0]):
            level_names = [pred_file[j] for pred_file in save_eval_names]
            level_eval = {eval_name: new_evaluation[eval_name] for eval_name in level_names}
            new_evaluation[f"AV_{common_prefix(level_names)}{common_suffix(level_names)}"] = average_metrics(level_eval)

    with open(args.evaluation_dict, "w") as ed:
        json.dump(new_evaluation, ed)