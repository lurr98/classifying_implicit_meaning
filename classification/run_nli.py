import torch, argparse, json, evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from prepare_data import load_impres_data, transform_inli_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer


def get_samples(item_file: str, context: bool):
    
    with open(item_file, "r") as itf:
        item_dict = json.load(itf)

    items = []
    for i, (sample_id, item) in enumerate(item_dict.items()):
        if context:
            context_before = "" if item['context_before'] == "no context" else item['context_before']
            context_after = "" if item['context_after'] == "no context" else item['context_after']
            items.append((sample_id, f"{context_before} {item['sentence_1']} {context_after}", f"{context_before} {item['sentence_2']} {context_after}"))
        else:
            items.append((sample_id, item['sentence_1'], item['sentence_2']))

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    return items


def run_nli_model(model_name: str, ph_pairs: list, out_file: str):

    config = AutoConfig.from_pretrained(model_name)  # or your model path
    print(config.id2label)

    exit()
    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if "deberta" in model_name: 
        labels = ["entailment", "not_entailment"]
    else:
        # The model outputs [contradiction, neutral, entailment]
        labels = ["contradiction", "neutral", "entailment"]
    predictions = {"labels": labels, "predictions": {}}

    for ph_pair in ph_pairs:
        idx, premise, hypothesis = ph_pair
        # Encode as a pair
        inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)

        # Get logits
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        predictions["predictions"][idx] = probs.tolist()

    
    with open(f"predictions/nli/{out_file}", "w") as of:
        json.dump(predictions, of)


def fine_tune_models(model: str, out_folder: str, dataset: dict, binary: bool, deberta: bool) -> None:

    def tokenize(batch):
        return tok(batch["premise"], batch["hypothesis"], truncation=True)

    def metrics(p):
        # preds = np.argmax(p.predictions, axis=1)
        # return {"acc": accuracy.compute(references=p.label_ids, predictions=preds)["accuracy"],
        #         "f1": f1.compute(references=p.label_ids, predictions=preds, average="weighted")["f1"]}

        # 1) unwrap
        logits = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
        # 2) ensure 2-D [N, C]
        import numpy as np
        logits = np.asarray(logits)
        if logits.ndim > 2:
            logits = logits.reshape(-1, logits.shape[-1])
        # 3) get class ids
        preds = logits.argmax(axis=-1)

        labels = p.label_ids
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels.reshape(-1)

        return {"acc": accuracy.compute(references=labels, predictions=preds)["accuracy"],
            "f1":  f1.compute(references=labels, predictions=preds, average="weighted")["f1"],}

    if binary:
        # (optional) Map string labels to ints, incl. 3→2 collapse
        # Example: entailment vs non-entailment (merge neutral+contradiction → 0, entailment → 1)
        # map neutral & contradiction to 1 and entailment to 0
        label_map = {2: 0, 1: 1, 0: 1}
        # if dataset["train"]["label"].dtype == object:
        dataset["train"]["label"] = dataset["train"]["label"].map(label_map)
        dataset["val"]["label"] = dataset["val"]["label"].map(label_map)

    train_ds = Dataset.from_pandas(dataset["train"], preserve_index=False)
    dev_ds = Dataset.from_pandas(dataset["val"], preserve_index=False)
    ds = DatasetDict(train=train_ds, validation=dev_ds)

    id2label = {0: "entailment", 1: "not_entailment"} if binary else {0: "contradiction", 2: "entailment", 1: "neutral"}
    label2id = {v: k for k, v in id2label.items()}

    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    ds = ds.map(tokenize, batched=True)

    # For 2-class training on roberta/bart MNLI checkpoints, this replaces their 3-class head.
    if binary:
        num_labels = 2  # set to 3 if you want the original MNLI setup
    else:
        num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels, id2label=id2label, label2id=label2id)

    collator = DataCollatorWithPadding(tokenizer=tok)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    args = TrainingArguments(
        output_dir="nli-ft",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=metrics
    )

    trainer.train()
    trainer.save_model(f"nli-ft/{out_folder}")
    tok.save_pretrained(f"nli-ft/{out_folder}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    # facebook/bart-large-mnli | roberta-large-mnli | MoritzLaurer/deberta-v3-large-zeroshot-v2
    parser.add_argument('model_name', type=str, help='name of the NLI model to be used')
    parser.add_argument('sample_file', type=str, help='name of the file containing the samples')
    parser.add_argument('out_file', type=str, help='name of the file containing the samples or the output folder for fine-tuned model')
    parser.add_argument('-ft', '--fine_tune', nargs='?', help='data to be fine-tuned on')
    parser.add_argument('-c', '--context', action='store_true')
    parser.add_argument('-b', '--binary', action='store_true')

    args = parser.parse_args()

    if args.fine_tune:
        if args.fine_tune == "imppres":
            ds = load_impres_data()
        else:
            ds = transform_inli_data(args.fine_tune.split(","))
        fine_tune_models(args.model_name, args.out_file, ds, args.binary, True if "deberta" in args.model_name else False)
    else:
        samples = get_samples(args.sample_file, args.context)
        run_nli_model(args.model_name, samples, args.out_file)