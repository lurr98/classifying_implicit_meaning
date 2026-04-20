import os, sys, random
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch, argparse, json, evaluate, random
import numpy as np
from datasets import Dataset, DatasetDict
from core.utils import read_json, write_json, load_impres_data, transform_inli_data
from fine_tuning import get_samples, build_huggingface_ds, fine_tune_models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

label2id = {"No": 0, "Yes": 1}
id2label = {0: "No", 1: "Yes"}


def run_nli_model(model_name: str, ph_pairs: list, out_file: str):

    config = AutoConfig.from_pretrained(model_name)  # or your model path
    print(config.id2label)

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if "deberta" in model_name or "fine-tuned" in model_name: 
        labels = ["entailment", "not_entailment"]
    else:
        # The model outputs [contradiction, neutral, entailment]
        labels = ["contradiction", "neutral", "entailment"]
    predictions = {"labels": labels, "predictions": {}}

    for ph_pair in ph_pairs:
        idx, premise, hypothesis = ph_pair
        # Encode as a pair
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

        # Get logits
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        predictions["predictions"][idx] = probs.tolist()

    
    write_json(predictions, out_file)
    print(f"Saved predictions in {out_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    # facebook/bart-large-mnli | roberta-large-mnli | MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    parser.add_argument('model_name', type=str, help='name of the NLI model to be used')
    parser.add_argument('data_splits', type=str, help='name of the directory containing the data splits')
    parser.add_argument('out_file', type=str, help='name of the file containing the samples or the output folder for fine-tuned model')
    parser.add_argument('-ft', '--fine_tune', action='store_true')
    parser.add_argument('-c', '--context', action='store_true')
    parser.add_argument('-b', '--binary', action='store_true')
    parser.add_argument('-ood', '--out_of_domain', action='store_true')

    args = parser.parse_args()

    base_dir = "/home/vault/v106be/v106be21/"
    models = {
        "facebook/bart-large-mnli": "bart", 
        "roberta-large-mnli": "roberta", 
        "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": "deberta",
        "fine_tuned_models/fine-tuned_bart_large_c": "bart",
        "fine_tuned_models/fine-tuned_roberta_large_c": "roberta",
        "fine_tuned_models/fine-tuned_deberta_large_c": "deberta"
        }

    data_splits = read_json(os.path.join(base_dir, args.data_splits, "splits.json"))
    samples = read_json(os.path.join(base_dir, args.data_splits, "samples.json"))
    if args.fine_tune:
        golds = read_json(os.path.join(base_dir, args.data_splits, "gold_standard.json"))
        train_samples = get_samples({idx: samples[idx] for idx in data_splits["train"]}, args.context, golds, nli=True)
        dev_samples = get_samples({idx: samples[idx] for idx in data_splits["dev"]}, args.context, golds, nli=True)
        fine_tune_models(train_samples, dev_samples, args.model_name, args.out_file, args.binary, nli=True, tune=True)
    else:
        if args.out_of_domain:
            test_samples = get_samples({idx: samples[idx] for idx in data_splits["out_of_domain_test"]}, args.context)
        else:
            test_samples = get_samples({idx: samples[idx] for idx in data_splits["test"]}, args.context)

        out_file = os.path.join(base_dir, "classifying_implicit_meaning", "predictions", "nli", "out_of_domain_test" if args.out_of_domain else "test", models[args.model_name], args.out_file)
        run_nli_model(args.model_name, test_samples, out_file)