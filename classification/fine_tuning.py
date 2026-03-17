import os, sys, random
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch, argparse, json, evaluate, random
import numpy as np
from datasets import Dataset, DatasetDict
from core.utils import read_json, write_json, load_impres_data, transform_inli_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

intro = " You are a classifier. Would altering the second text by inserting the text in angle brackets meaningfully change most readers understand the text? Answer with Yes or No.\n\n"

label2id = {"No": 0, "Yes": 1}
id2label = {0: "No", 1: "Yes"}


def get_samples(item_dict: dict, context: bool, gold: dict={}, nli: bool=False):

    items = []
    for i, (sample_id, item) in enumerate(item_dict.items()):
        if context or nli is False:
            context_before = "" if item['context_before'] == "no context" else item['context_before']
            context_after = "" if item['context_after'] == "no context" else item['context_after']
            item_1 = f"{context_before} {item['sentence_1']} {context_after}"
            item_2 = f"{context_before} {item['sentence_2']} {context_after}"
            if not nli:
                item_1 = f"First text: {item_1}"
                item_2 = f"Second text: {item_2}"
        else:
            item_1 = item['sentence_1']
            item_2 = item['sentence_2']
        
        if gold:
            label = gold[sample_id]["MACE Label"]
            items.append((sample_id, item_1, item_2, label))
        else:
            items.append((sample_id, item_1, item_2))

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    return items


def build_huggingface_ds(train_samples: list, dev_samples: list, nli: bool):

    def build_split_ds(samples: list, nli: bool=False):

        ds_dict = {
            "id": [sample[0] for sample in samples],
        }
        if nli:
            ds_dict["premise"] = [sample[1] for sample in samples]
            ds_dict["hypothesis"] = [sample[2] for sample in samples]
            ds_dict["labels"] = [label2id[sample[3]] for sample in samples]

        else:
            ds_dict["prompt"] = [intro + sample[1] + sample[2] for sample in samples]
            ds_dict["labels"] = [sample[3] for sample in samples]

        ds = Dataset.from_dict(ds_dict)
        return ds

    train_ds = build_split_ds(train_samples, nli)
    dev_ds = build_split_ds(dev_samples, nli)

    ds = DatasetDict({
        "train": train_ds,
        "dev": dev_ds
    })

    return ds


def tokenize_causal(batch, tokenizer, max_length=512):
    prompts = batch["prompt"]
    labels = batch["labels"]

    full_texts = [p + " " + l for p, l in zip(prompts, labels)]

    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    labels_ids = torch.tensor(tokenized["input_ids"])

    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length
        )["input_ids"]

        prompt_len = min(len(prompt_ids), labels_ids.size(1) - 1)
        labels_ids[i, :prompt_len] = -100

    tokenized["labels"] = labels_ids
    return tokenized



# def fine_tune_models(
#     train_samples: list,
#     dev_samples: list,
#     model_name: str,
#     out_folder: str,
#     binary: bool,
#     nli: bool,
#     seed: int = 42,
# ):
#     # -------------------------------
#     # Tokenization functions
#     # -------------------------------
#     def tokenize_nli(batch):
#         return tokenizer(
#             batch["premise"],
#             batch["hypothesis"],
#             truncation=True,
#             padding=False,
#         )
# 
#     def tokenize_instruction(batch):
#         return tokenizer(
#             batch["prompt"],
#             truncation=True,
#             padding=False,
#         )
# 
#     # -------------------------------
#     # Metrics
#     # -------------------------------
#     def metrics(eval_pred):
#         logits, labels = eval_pred
#         if isinstance(logits, tuple):
#             logits = logits[0]
#         preds = np.argmax(logits, axis=-1)
#         return metric.compute(predictions=preds, references=labels)
# 
#     # -------------------------------
#     # Reproducibility
#     # -------------------------------
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# 
#     # -------------------------------
#     # Dataset
#     # -------------------------------
#     ds = build_huggingface_ds(train_samples, dev_samples, nli)
# 
#     num_labels = 2 if binary else 3
# 
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         # model.config.pad_token_id = tokenizer.pad_token_id
# 
# 
#     # -------------------------------
#     # Model + encoding
#     # -------------------------------
#     if nli:
#         # === NLI MODELS (BART / DeBERTa / RoBERTa) ===
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=num_labels,
#             id2label=id2label,
#             label2id=label2id,
#             ignore_mismatched_sizes=True,
#         )
# 
#         model.config.use_cache = False
#         encoded_ds = ds.map(tokenize_nli, batched=True)
# 
#     else:
#         # === INSTRUCTION MODELS (LLaMA / Mixtral / Mistral) ===
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=num_labels,
#             id2label=id2label,
#             label2id=label2id,
#             ignore_mismatched_sizes=True,
#             # device_map="auto",
#             trust_remote_code=True,
#         )
# 
#         # memory-safe finetuning
#         model = prepare_model_for_kbit_training(model)
#         model.config.use_cache = False
#         model.gradient_checkpointing_enable()
# 
#         model = get_peft_model(model, lora_config)
# 
#         encoded_ds = ds.map(tokenize_instruction, batched=True)
# 
#     # -------------------------------
#     # Logging sanity checks
#     # -------------------------------
#     print(f"Train size: {len(encoded_ds['train'])}")
#     print(f"Dev size: {len(encoded_ds['dev'])}")
#     print(f"Labels: {model.config.num_labels}")
# 
#     # -------------------------------
#     # Trainer setup
#     # -------------------------------
#     collator = DataCollatorWithPadding(tokenizer)
#     metric = evaluate.load("accuracy")
# 
#     args = TrainingArguments(
#         output_dir=out_folder,
#         learning_rate=2e-5,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=4,
#         gradient_accumulation_steps=8,
#         num_train_epochs=1,  # IMPORTANT for instruction models
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="accuracy",
#         greater_is_better=True,
#         warmup_ratio=0.1,
#         weight_decay=0.01,
#         # bf16=True,
#         logging_steps=50,
#         save_total_limit=2,
#         report_to="none",
#     )
# 
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=encoded_ds["train"],
#         eval_dataset=encoded_ds["dev"],
#         tokenizer=tokenizer,
#         data_collator=collator,
#         compute_metrics=metrics,
#     )
# 
#     # -------------------------------
#     # Train & save
#     # -------------------------------
#     trainer.train()
#     trainer.save_model(out_folder)
#     tokenizer.save_pretrained(out_folder)
# 
#     print(f"Saved model to {out_folder}")

def fine_tune_models(train_samples: list, dev_samples: list, model: str, out_folder: str, binary: bool, nli: bool, seed: int=42) -> None:


    def tokenize_nli(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], max_length=265, truncation=True)

    def tokenize_cl(batch):
        return tokenizer(batch["prompt"], truncation=True, padding=False)

    # def metrics(eval_pred):
    # 
    #     logits, labels = eval_pred
# 
    #     # get logits tensor from tuple
    #     if isinstance(logits, tuple):
    #         logits = logits[0]
# 
    #     predictions = np.argmax(logits, axis=-1)
# 
    #     return metric.compute(predictions=predictions, references=labels)


    def metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="macro")

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ds = build_huggingface_ds(train_samples, dev_samples, nli)

    # For 2-class training on roberta/bart MNLI checkpoints, this replaces their 3-class head.
    if binary:
        num_labels = 2  # set to 3 if you want the original MNLI setup
    else:
        num_labels = 3

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    # tokenizer.pad_token = tokenizer.eos_token
    
    if nli:
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        encoded_ds = ds.map(tokenize_nli, batched=True)
        model.config.use_cache = False

        for p in model.model.decoder.parameters():
            p.requires_grad = False


    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )   
        # Required for stable k-bit training
        model = prepare_model_for_kbit_training(model)

        # Reduce activation memory
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # Attach LoRA adapters
        model = get_peft_model(model, lora_config)
        encoded_ds = ds.map(tokenize_cl, batched=True)

    print(f"Number of training samples: {len(encoded_ds['train']['id'])}")
    print(f"Model num_labels: {model.config.num_labels}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    # metric = evaluate.load("accuracy")
    metric = evaluate.load("f1")

    # args = TrainingArguments(
    #     output_dir=out_folder,
    #     learning_rate=1e-5,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=8,
    #     gradient_accumulation_steps=8,  # effective = 32
    #     num_train_epochs=3,
# 
    #     label_smoothing_factor=0.1,
# 
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    #     greater_is_better=True,
# 
    #     warmup_ratio=0.1,
    #     weight_decay=0.01,
    #     bf16=True,
# 
    #     logging_steps=50,
    #     save_total_limit=1,
    #     dataloader_num_workers=4,
    #     report_to="none"
    # )

    # BART BEST
    args = TrainingArguments(
        output_dir=out_folder,

        learning_rate=3e-6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch = 16

        num_train_epochs=3,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.0,

        bf16=True,
        logging_steps=50,
        save_total_limit=1,
        dataloader_num_workers=4,
        report_to="none"
    )



    # ROBERTA & DEBRTA BEST
    # args = TrainingArguments(
    #     output_dir=out_folder,
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=8,
    #     num_train_epochs=5,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="accuracy",  # Use accuracy to select best model
    #     greater_is_better=True,  # Higher accuracy is better
    #     lr_scheduler_type="linear",  # Already good; decays LR linearly
    #     warmup_ratio=0.1,  # Warm up LR for 10% of steps to stabilize training
    #     weight_decay=0.01,  # L2 regularization to prevent overfitting
    #     # gradient_checkpointing=True,
    #     fp16=False,         # switch to True only if stable
    #     bf16=True, 
    #     label_names=["labels"],  # As-is
    #     logging_steps=50,  # Log every 50 steps for monitoring
    #     save_total_limit=2,  # Keep only 2 best checkpoints to save space
    #     dataloader_num_workers=4,  # Speed up data loading (adjust based on CPU cores)
    #     report_to="none"  # Disable wandb/tensorboard unless you want logging
    # )
# 
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["dev"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()
    trainer.save_model(out_folder)
    tokenizer.save_pretrained(out_folder)
    print(f"Saved fine-tuned model in {out_folder}")
