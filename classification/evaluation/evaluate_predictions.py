import argparse
import json
import re
import os
from plot_script import plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
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
                pattern = rf"\b{label}\b"
                matches.extend(re.findall(pattern, pred, re.IGNORECASE))
                if len(list(set(matches))) == 1:
                    # return matches[0]
                    return label
            else:
                print(f"Unclear Label in Experiment {exp_name}: {pred}")
                return "UNK"


def evaluate_predictions_classification(gold_labels: dict, predictions: dict, evaluation_dict: dict, key: str, plot_name: str, second_pred: bool, mistakes: Union[bool, str], gold_anno: str="MACE Label") -> dict:

    pred_list, gold_list, id_list = [], [], []
    for id_sample, sample in gold_labels.items():
        if id_sample in predictions:
            gold_list.append(sample[gold_anno])
            pred_list.append(get_prediction(predictions[id_sample], key))
            id_list.append(id_sample)
    
    if gold_list == []:
        print(f"Topic {key} is empty, skipping evaluation.")
        return

    if mistakes:
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


def evaluate_predictions_nli(gold_labels: dict, predictions: dict, evaluation_dict: dict, key: str, plot_name: str, second_pred: str, mistakes: Union[bool, str], gold_anno: str="MACE Label") -> dict:

    if len(list(predictions["predictions"].values())[0]) == 2:
        # TODO: not_entailment could also be "Tie"
        label_dict = {"not_entailment": "Yes", "entailment": "No", "contradiction": "Yes", "neutral": "Yes"}
    else:
        label_dict = {"contradiction": "Yes", "neutral": "Tie", "entailment": "No"}
    pred_list, gold_list, id_list = [], [], []
    if second_pred:
        plot_name = plot_name[:-4] + f"_secondprob{second_pred}.png"
    for id_sample, sample in gold_labels.items():
        if id_sample in predictions["predictions"]: 
            gold_list.append(sample[gold_anno])
            preds = predictions["predictions"][id_sample]
            id_list.append(id_sample)
            if second_pred and len(preds) > 2:
                if preds[0] > float(second_pred) or preds[2] > float(second_pred):
                    binary_preds = [preds[0], preds[2]]
                    pred_list.append(predictions["labels"][preds.index(max(binary_preds))])
                else:
                    pred_list.append(predictions["labels"][preds.index(max(preds))])
            else:
                pred_list.append(predictions["labels"][preds.index(max(preds))])

    pred_list = [label_dict[item] for item in pred_list]

    if gold_list == []:
        print(f"Topic {key} is empty, skipping evaluation.")
        return

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

    try:
        with open(mistakes, "r") as m:
            mistake_dict = json.load(m)
    except FileNotFoundError:
        mistake_dict = {}

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

    return mistake_dict



def get_clear_cases(gold_standard: dict) -> dict:

    clear_cases = {}
    for sample_id, sample in gold_standard.items():
        if sample["Label Distribution"][0] == sum(sample["Label Distribution"]) or sample["Label Distribution"][1] == sum(sample["Label Distribution"]):
            clear_cases[sample_id] = sample

    return clear_cases


def get_topic_cases(gold_standard: dict, id2topic: dict, topic: str) -> dict:

    topic_cases = {}
    for sample_id, sample in gold_standard.items():
        if id2topic[sample_id] == topic:
            topic_cases[sample_id] = sample

    return topic_cases


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
    parser.add_argument('preds_dir', type=str, help='name of the folder containing the predictions')
    parser.add_argument('evaluation_dict', type=str, help='name of the evaluation file')
    parser.add_argument('-m', '--mistake', nargs='?', help='name of the mistake file if needed')
    parser.add_argument('-sp', '--second_pred', nargs='?', help='threshold for second prediction')
    parser.add_argument('-nli', '--nli', action='store_true')
    parser.add_argument('-p', '--plot_name', action='store_true')
    parser.add_argument('-cc', '--clear_cases', action='store_true')
    parser.add_argument('-t', '--topics', action='store_true')

    args = parser.parse_args()

    base_dir = "/home/vault/v106be/v106be21/"
    ood = "out_of_domain_test" if "out_of_domain_test" in args.preds_dir else "test"

    with open(os.path.join(base_dir, "implicit_data", "id2topic.json"), "r") as jsn:
        id2topic = json.load(jsn)

    print("Read targets.")
    with open(os.path.join(base_dir, args.targets), "r") as trg:
        targets = json.load(trg)

    if args.nli:
        evaluate = evaluate_predictions_nli
        nli = "nli"
    else:
        evaluate = evaluate_predictions_classification
        nli = "classification"

    try:
        with open(os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", nli, args.evaluation_dict), "r") as ev:
            evaluation = json.load(ev)
    except FileNotFoundError:
        evaluation = {}
    
    save_eval_names = [[] for _ in os.listdir(os.path.join(base_dir, "classifying_implicit_meaning", "predictions", nli, args.preds_dir))]
    
    mistake_path = os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", nli, args.mistake) if args.mistake else False 


    for i, file in enumerate(os.listdir(os.path.join(base_dir, "classifying_implicit_meaning", "predictions", nli, args.preds_dir))):

        if not file.endswith(".json"):
            continue
            
        with open(os.path.join(base_dir, "classifying_implicit_meaning", "predictions", nli, args.preds_dir, file), "r") as preds:
            predictions = json.load(preds)

        if args.second_pred:
            evaluation_name = "_".join(file.split("predictions_")[1:])[:-5] + f"_secondprob{args.second_pred}"
        else:
            evaluation_name = "_".join(file.split("predictions_")[1:])[:-5]
    
        save_eval_names[i].append(evaluation_name)

        if args.clear_cases:
            cc_targets = get_clear_cases(targets)
            cc_evaluation_name = evaluation_name + "_cc"
            save_eval_names[i].append(cc_evaluation_name)
    
            new_evaluation = evaluate(cc_targets, predictions, evaluation, cc_evaluation_name, os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", "plots", f"cm_{ood}_{cc_evaluation_name}.png") if args.plot_name else "", args.second_pred, mistake_path)
    
        if args.topics:
            for topic in set(id2topic.values()):
                topic_targets = get_topic_cases(targets, id2topic, topic)
                topic_evaluation_name = evaluation_name + f"_{topic}"
                save_eval_names[i].append(topic_evaluation_name)
     
                new_evaluation = evaluate(topic_targets, predictions, evaluation, topic_evaluation_name, os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", "plots", f"cm_{ood}_{topic_evaluation_name}_{args.preds_dir.split('/')[-1]}" + ".png") if args.plot_name else "", args.second_pred, mistake_path)
                if new_evaluation:
                    print(f"Saving to {os.path.join(base_dir, 'classifying_implicit_meaning', 'evaluation', ood, args.evaluation_dict)}")
                    with open(os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", ood, args.evaluation_dict), "w") as ed:
                        json.dump(new_evaluation, ed)
    
        new_evaluation = evaluate(targets, predictions, evaluation, evaluation_name, os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", "plots", f"cm_{ood}_{evaluation_name}_{args.preds_dir.split('/')[-1]}.png") if args.plot_name else "", args.second_pred, mistake_path)

    print(f"Saving to {os.path.join(base_dir, 'classifying_implicit_meaning', 'evaluation', ood, args.evaluation_dict)}")
    # if evaluation file exists, update it, otherwise create it
    with open(os.path.join(base_dir, "classifying_implicit_meaning", "evaluation", ood, args.evaluation_dict), "w") as ed:
        json.dump(new_evaluation, ed)