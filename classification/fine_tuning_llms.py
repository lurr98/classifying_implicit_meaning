import os, sys, random
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch, argparse, json, evaluate, random
import numpy as np
from collections import Counter
from datasets import Dataset, DatasetDict
from core.utils import read_json, write_json, load_impres_data, transform_inli_data
from trl import SFTConfig, SFTTrainer
from transformers import (pipeline, AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate

# The `evaluate` library ships with a ready‑made accuracy metric
accuracy_metric = evaluate.load("accuracy")


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

prompt = "Would altering the second text by inserting the text in angle brackets meaningfully change most readers understand the text? Answer with Yes or No.\n"


def _extract_yes_no(text: str) -> str:
    """
    Return a normalized 'yes' or 'no' string if it appears at the start
    of the generated answer; otherwise return an empty string.
    """
    text = text.strip()
    if text.startswith("Yes"):
        return "Yes"
    if text.startswith("No"):
        return "No"
    # fallback – look for the first token that matches
    for token in text.split():
        if token in ("Yes", "No"):
            return token
    return ""   # will be counted as wrong


def compute_metrics(eval_pred):
    preds_ids, label_ids = eval_pred          # label_ids comes from the "label" column
    pred_texts = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

    # Normalise predictions to "yes"/"no"
    pred_norm = [_extract_yes_no(t) for t in pred_texts]

    # `label_ids` already contains the lower‑cased "yes"/"no"
    true_norm = label_ids

    return {"accuracy": accuracy_metric.compute(predictions=pred_norm,
                                                references=true_norm)["accuracy"]}


def upsample_minority(samples, label_index=3, seed=42):
    random.seed(seed)

    # group by label
    by_label = {}
    for s in samples:
        by_label.setdefault(s[label_index], []).append(s)

    counts = Counter(s[label_index] for s in samples)
    max_count = max(counts.values())

    upsampled = []
    for label, group in by_label.items():
        if len(group) < max_count:
            extra = random.choices(group, k=max_count - len(group))
            upsampled.extend(group + extra)
        else:
            upsampled.extend(group)

    random.shuffle(upsampled)
    return upsampled


def build_huggingface_ds(train_samples: list, dev_samples: list):

    def build_split_ds(samples: list, nli: bool=False):

        ds_dict = {
            "id": [sample[0] for sample in samples],
            "label": [sample[3].strip() for sample in samples],
        }
        messages = [[
            {"role": "system", "content": "You are an annotator that has to annotate the data. Only respond with one of these labels: Yes or No."},
            {"role": "user", "content": prompt + sample[1] + sample[2]},
            {"role": "assistant", "content": sample[3]}

        ]
        for sample in samples]

        ds_dict["messages"] = messages

        ds = Dataset.from_dict(ds_dict)
        return ds

    train_ds = build_split_ds(train_samples)
    dev_ds = build_split_ds(dev_samples)

    ds = DatasetDict({
        "train": train_ds,
        "dev": dev_ds
    })

    return ds


def fine_tune_instruction_lm(train_samples,
    dev_samples,
    model_name,
    out_folder,
    seed=42
    ):

    # ---- seeds ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)

    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # ---- dataset ----

    # train_samples = upsample_minority(train_samples)

    ds = build_huggingface_ds(train_samples, dev_samples)

    print(ds["train"][0])

    # Configure the SFTTrainer
    sft_config = SFTConfig(
        output_dir=out_folder,
        load_best_model_at_end=True,
        completion_only_loss=True,
        label_smoothing_factor=0.1,
        metric_for_best_model="accuracy",   # or "accuracy"
        greater_is_better=True,      
        num_train_epochs=3,
        save_total_limit=1,                  # keep only 1 checkpoint
        eval_strategy="steps",               # or "epoch"
        save_strategy="steps",                # or "epoch"
        eval_steps=500,
        save_steps=500,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # max_steps=1000,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
          # Set according to your GPU memory capacity
        learning_rate=5e-6,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        # save_steps=100,  # Frequency of saving model checkpoints
        # eval_strategy="steps",  # Evaluate the model at regular intervals
        # eval_steps=50,  # Frequency of evaluation
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        eval_dataset=ds["dev"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_folder)
    tokenizer.save_pretrained(out_folder)

    print(f"Saved instruction-tuned classifier to {out_folder}")