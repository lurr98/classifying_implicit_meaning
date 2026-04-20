import os
cache_dir = os.path.join("/home/vault/v106be/v106be21", '.cache')

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['HF_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse, json, torch, spacy, pickle, re, sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter
from huggingface_hub import login
from fine_tuning import get_samples
from fine_tuning_llms import fine_tune_instruction_lm
from transformers import (pipeline, AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig)
from typing import Union
# feature importances
from interpret_models import * 

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from core.utils import read_json, write_json, format_few_examples, format_target_sample, prepare_linear_classifier

torch.cuda.empty_cache()

load_dotenv()
login(token=os.getenv("ACCESS_TOKEN"))

# prompt = "You are an annotator that has to annotate the data based on the following instruction:\n\n<break>\n\n"
labels = ["No", "Yes"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True
)

## BASELINES
    
def majority_baseline(item_dict: dict, majority_label: str) -> dict:

    predictions = {}

    for i, (sample_id, item) in enumerate(item_dict.items()):
        
        predictions[sample_id] = majority_label
        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))  

    return predictions


def adj_baseline(item_dict: dict, adj_label: str, not_adj_label: str, x: int) -> dict:

    nlp = spacy.load("en_core_web_sm")

    def check_for_adjs(revision: str):
        doc = nlp(revision)
        tokens = [token for token in doc if not token.is_punct and not token.is_space]

        pos_counts = Counter(token.pos_ for token in tokens)
        total = sum(pos_counts.values())

        for pos, count in pos_counts.items():
            if pos == "ADJ":
                # if # of ADJ > x
                if count >= x:
                    return True
        return False

    predictions = {}

    for i, (sample_id, item) in enumerate(item_dict.items()):
        if check_for_adjs(item["sentence_2"]):
            predictions[sample_id] = adj_label
        else:
            predictions[sample_id] = not_adj_label

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))  

    return predictions


## LLMs

def classify_samples_gpt(item_dict: dict, instructions: str, model: str, api_file: str, out_path: str) -> dict:

    api_key = api_file[0].split(' ')[-1].strip()
    print(api_key)

    predictions = {}

    client = OpenAI(api_key=api_key)

    for i, (sample_id, item) in enumerate(item_dict.items()):
        formatted_item = format_target_sample(item)


        if i == 0:
            print(instructions)
        print(formatted_item)
        print("-----")

        message = (
            instructions
            + f"\n\n{formatted_item}"
        )

        system_str = f"Start your reply with exactly one of the following labels: {' or '.join(labels)}."
        messages = [{"role": "system", "content": system_str}, {"role": "user", "content": message}]

        chat = client.chat.completions.create(
            model=model,
            temperature=0.0,
            top_p=1.0,
            messages=messages
        )
        reply = chat.choices[0].message.content
        
        predictions[sample_id] = reply

        write_json(predictions, out_path)

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    return predictions


def classify_samples_oss_gpt(item_dict: dict, instructions: str, model, api_file: str, out_path: str) -> dict:

    predictions = {}

    for i, (sample_id, item) in enumerate(item_dict.items()):

        print("Going through samples now…")
        formatted_item = format_target_sample(item)

        message = (
            instructions
            + f"\n\n{formatted_item}"
        )

        system_str = f"Start your reply with exactly one of the following labels: {' or '.join(labels)}"
        messages = [{"role": "system", "content": system_str}, {"role": "user", "content": message}]
        print("Going into pipe now…")
        response = model(
            messages,
            temperature=0.01,
            top_p=1,
            use_cache=True,
            return_full_text=False,
            max_new_tokens=1024,
        )

        reply = response[0]["generated_text"]

        print("Extracted reply…")

        found = re.findall("assistantfinal(No|Yes)", reply)
        if found:
            predictions[sample_id] = found[0]
        else:
            predictions[sample_id] = "NO_LABEL"

        write_json(predictions, out_path)
        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


def classify_samples_deepseek(item_dict: dict, instructions: str, model: str, api_file: str, out_path: str) -> dict:

    api_key = api_file[1].split(" ")[-1].strip()

    predictions = {}
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for i, (sample_id, item) in enumerate(item_dict.items()):
        formatted_item = format_target_sample(item)

        message = (
            instructions
            + f"\n\n{formatted_item}"
        )
        messages = [{"role": "system", "content": "You are an annotator that has to annotate the data based on the following instruction. Only respond with one of these labels: Yes or No." + instructions + f"Start your reply with exactly one of the following labels: {' or '.join(labels)}."}, {"role": "user", "content": formatted_item}]
        
        print(messages)
        chat = client.chat.completions.create(
            model="deepseek-reasoner", # deepseek-chat
            temperature=0.01,
            messages=messages
        )
        reply = chat.choices[0].message.content
        predictions[sample_id] = reply

        write_json(predictions, out_path)

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


# def classify_samples_huggingface(item_dict: dict, instructions: str, model_tokenizer: tuple, api_file: str, out_path: str) -> dict:
def classify_samples_huggingface(item_dict: dict, instructions: str, pipe, api_file: str, out_path: str) -> dict:

    def parse_output(output: str) -> str:
        for label in labels:
            pattern = rf"\b{label}\b"
            match = re.findall(pattern, output, re.IGNORECASE)
            if match:
                return label
        print(output)
        return None
    # model, tokenizer = model_tokenizer
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model.to(device)

    predictions = {}

    for i, (sample_id, item) in enumerate(item_dict.items()):
        print(i)
        formatted_item = format_target_sample(item)

        if i == 0:
            print(instructions)
            print(formatted_item)
            print("-----")
        
        prompt =  instructions 
        prompt += formatted_item 
        prompt += f"\nStart your reply with exactly one of the following labels: {' or '.join(labels)}.\nLabel:"

            
        messages = [
            {"role": "system", "content": "You are an annotator that has to annotate the data based on the following instruction. Only respond with one of these labels: Yes or No."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Generate the response using the pipeline
            output = pipe(messages, max_new_tokens=5)
            response = output[0]['generated_text'][-1]["content"]
            print(f"Response from model: {response}")

            # First try to extract JSON from the response
            response_data = parse_output(response)

            # If the JSON is invalid, attempt to regenerate the response
            if response_data is None:
                print("First attempt failed, regenerating response...")
                output = pipe(messages, max_new_tokens=5)
                response = output[0]['generated_text'][-1]["content"]
                print(f"Response from model (regenerated): {response}")
                response_data = parse_output(response)

            # If both attempts fail, skip this sample
            if response_data is None:
                print(f"Invalid response for sample {sample_id}.")
                reply = "NO_LABEL"
            else:
                reply = response_data

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            return None
        # inputs = tokenizer(
        #     prompt,
        #     return_tensors="pt"
        # ).to(device)
# 
        # with torch.no_grad():
        #     outputs = model(
        #         **inputs,
        #         max_new_tokens=5,
        #         use_cache=True
        #     )
        #     # Slice off the prompt tokens
        #     generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
# 
        # # Decode ONLY the new tokens
        # reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        predictions[sample_id] = reply

        write_json(predictions, out_path)

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


def classify_samples_mixtral(item_dict: dict, instructions: str, model_tokenizer: tuple, api_file: str, out_path: str) -> dict:

    def parse_output(output: str) -> str:
        for label in labels:
            pattern = rf"\b{label}\b"
            match = re.findall(pattern, output, re.IGNORECASE)
            if match:
                return label
        print(output)
        return None
    model, tokenizer = model_tokenizer
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model.to(device)

    predictions = {}

    # sent_ex = ["I absolutely loved this movie. It was fantastic and enjoyable from start to finish.", "This was a terrible experience. Everything was broken and nothing worked properly.", "Good.", "Bad."]

    for i, (sample_id, item) in enumerate(item_dict.items()):
        print(i)
        formatted_item = format_target_sample(item)

        if i == 0:
            print(instructions)
            print(formatted_item)
            print("-----")
        
        prompt =  instructions 
        prompt += formatted_item 
        prompt += f"\nStart your reply with exactly one of the following labels: {' or '.join(labels)}.\nLabel:"
        # prompt = "Start your reply with exactly one of the following labels: Positive or Negative.\nLabel:"

            
        messages = [
            {"role": "system", "content": "You are an annotator that has to annotate the data based on the following instruction. Only respond with one of these labels: Yes or No."},
            {"role": "user", "content": prompt}
        ]

        # messages = [
        #     {"role": "system", "content": "You are a sentiment classifier. Is the sentiment of the following text Positive or Negative? Answer with Positive or Negative."},
        #     {"role": "user", "content": sent_ex[i]}
        # ]
        

        # Generate the response using the pipeline
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
# 
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=5,
                use_cache=True,
                temperature=0.01
            )
            # Slice off the prompt tokens
            reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[-1]
            # reply = tokenizer.decode(outputs, skip_special_tokens=True)[0]
            print(f"Response from model: {reply}")
# 
        # # Decode ONLY the new tokens
        # reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # First try to extract JSON from the response
        response_data = parse_output(reply)

        # If the JSON is invalid, attempt to regenerate the response
        if response_data is None:
            print("First attempt failed, regenerating response...")
            outputs = model.generate(
                inputs,
                max_new_tokens=5,
                use_cache=True,
                temperature=0.01
            )
            # Slice off the prompt tokens
            reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("[/INST]")[-1]
            print(f"Response from model (regenerated): {reply}")
            response_data = parse_output(reply)

        # If both attempts fail, skip this sample
        if response_data is None:
            print(f"Invalid response for sample {sample_id}.")
            reply = "NO_LABEL"
        else:
            reply = response_data
        

        predictions[sample_id] = reply

        write_json(predictions, out_path)

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify a single study from topic_data')
    parser.add_argument('--data_split', type=str, help='the data split we want to test on ("test" or "out_of_domain_test")')
    parser.add_argument('--api', type=str, help='name of the file containing API key')
    parser.add_argument('--model', type=str, help='name of the model to be used')
    parser.add_argument('--config', type=str, help='name of the trainer config to be used')
    parser.add_argument('--lora_config', type=str, help='name of the lora config to be used')
    # parser.add_argument('outfile', type=str, help='name of the output file containing the predictions')
    parser.add_argument('-f', '--few_shot', nargs='?', help='number of few shot examples that will be presented to the model, split by comma if multiple')
    parser.add_argument('-z', '--zero_shot', action='store_true')
    parser.add_argument('-ft', '--fine_tuning', action='store_true')
    parser.add_argument('-mj', '--majority_bs', action='store_true')

    args = parser.parse_args()

    model_names = {
        "openai/gpt-oss-20b": "oss_GPT",
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral", 
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama", 
        "meta-llama/Meta-Llama-3-8B": "llama", 
        "classification/fine_tuned_models/llama": "llama", 
        "classification/fine_tuned_models/qwen": "qwen", 
        "classification/fine_tuned_models/mistral": "mistral", 
        "Qwen/Qwen3-4B-Instruct-2507": "qwen"}

    # ft_base = {
    #     "ft_0402/llama": "meta-llama/Meta-Llama-3-8B-Instruct", 
    #     "ft_0902/llama": "meta-llama/Meta-Llama-3-8B-Instruct", 
    #     "ft_0402_depr/llama": "meta-llama/Meta-Llama-3-8B-Instruct", 
    #     "ft_0402_depr2/llama": "meta-llama/Meta-Llama-3-8B-Instruct", 
    #     "ft_0402/qwen": "Qwen/Qwen3-4B-Instruct-2507", 
    #     "ft_0902/qwen": "Qwen/Qwen3-4B-Instruct-2507", 
    #     "ft_0402_depr/qwen": "Qwen/Qwen3-4B-Instruct-2507", 
    #     "ft_0402_depr2/qwen": "Qwen/Qwen3-4B-Instruct-2507", 
    #     "ft_0402/mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "ft_0902/mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "ft_0402_depr/mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "ft_0402_depr2/mistral": "mistralai/Mistral-7B-Instruct-v0.3"}

    print("Is this even running?")
    base_dir = "/home/vault/v106be/v106be21/"

    # Load samples from the study's samples folder
    samples = read_json(os.path.join(base_dir, "implicit_data", "samples.json"))
    data_splits = read_json(os.path.join(base_dir, "implicit_data", "splits.json"))
    
    if args.fine_tuning:
        golds = read_json(os.path.join(base_dir, "implicit_data", "gold_standard.json"))
        train_samples = get_samples({idx: samples[idx] for idx in data_splits["train"]}, context=True, gold=golds, nli=True)
        dev_samples = get_samples({idx: samples[idx] for idx in data_splits["dev"]}, context=True, gold=golds, nli=True)
        fine_tune_instruction_lm(train_samples, dev_samples, args.model, args.config, args.lora_config, seed=42, tune=True)
        
    else:
        split_samples = {idx: samples[idx] for idx in data_splits[args.data_split]}

        if args.majority_bs:
            model_name = "mj_baseline"

        elif "gpt" in args.model:
            with open(args.api, "r") as af:
                api_keys = af.readlines()

            if args.model == "oss_gpt":
                model_name = "oss_GPT"
                model = pipeline(
                    "text-generation",
                    model="openai/gpt-oss-20b",
                    dtype=torch.bfloat16,
                    device_map="cuda"
                   )
                classify = classify_samples_oss_gpt 
            else:
                model_name = "gpt"
                model = args.model
                classify = classify_samples_gpt 
        elif args.model == "deepseek":
            with open(args.api, "r") as af:
                api_keys = af.readlines()
            model_name = args.model
            model = args.model
            classify = classify_samples_deepseek
        elif "mixtral" in args.model.lower():
            model_name = model_names[args.model]
            tokenizer = AutoTokenizer.from_pretrained(args.model, token="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")
            pre_model = AutoModelForCausalLM.from_pretrained(
                                                    args.model, 
                                                    device_map="auto", 
                                                    quantization_config=bnb_config)

            print("Loaded Mistral model and tokenizer…")
            model = (pre_model, tokenizer)
            api_keys = ""
            classify = classify_samples_mixtral
        else:
            # if "ft" in args.model:
            #     base_name = ft_base[args.model]
            #     model_name = model_names[base_name]
            #     tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
# 
            #     model = AutoModelForCausalLM.from_pretrained(
            #         base_name,
            #         device_map="auto",          # puts layers on available GPUs/CPU automatically
            #         # load_in_8bit=True,         # optional, matches your training config
            #         trust_remote_code=True
            #     )
# 
            #     # Attach the LoRA weights
            #     full_model = PeftModel.from_pretrained(model, args.model)
            # else:
            full_model = args.model
            model_name = model_names[args.model]
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(full_model, use_fast=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # tokenizer = AutoTokenizer.from_pretrained(args.model, token="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")
            
            model = pipeline(
                "text-generation",
                model=full_model,
                tokenizer=tokenizer,
                # device="cuda",
                torch_dtype=torch.bfloat16,
                # quantization_config=bnb_config,
                token=os.getenv("ACCESS_TOKEN")
,
                temperature=0.01
            )
            print("Loaded model and tokenizer…")
            api_keys = ""
            classify = classify_samples_huggingface

        out_dir = os.path.join(base_dir, "classifying_implicit_meaning", 'predictions', "classification", args.data_split, model_name)
        os.makedirs(out_dir, exist_ok=True)

        print("Here now…")
        if args.few_shot:
            print("Loading few shot samples…")
            few_shot_samples = read_json(os.path.join(base_dir, "implicit_data", "few_shots.json"))
            with open(os.path.join(base_dir, "classifying_implicit_meaning", "instructions", "instructions_few.md"), "r") as i:
                    intro = i.read() 

            for few in args.few_shot.split(","):
            # choose the few_shots file from the study
                print(f"Working on few shot {few}…")

                intro = format_few_examples(few_shot_samples, samples, intro, few)

                print("Classifying…")
                out_path = os.path.join(out_dir, f"predictions_few{few}_updintro.json")
                preds = classify(split_samples, intro, model, api_keys, out_path)

                print(f"Saved predictions to {out_path}")

        elif args.zero_shot:
            if args.majority_bs:
                preds = majority_baseline(split_samples, "No")
                out_path = os.path.join(out_dir, "predictions.json")
                write_json(preds, out_path)
                print(f"Saved majority baseline predictions to {out_path}")

            else:
                with open(os.path.join(base_dir, "classifying_implicit_meaning", "instructions", "instructions_zero.md"), "r") as i:
                    intro = i.read()

                print("Going into classification…")
                if "ft" in args.model:
                    ft_tag = args.model.split("/")[0]
                    out_path = os.path.join(out_dir, f"predictions_{ft_tag}.json")
                    intro = "Would altering the second text by inserting the text in angle brackets meaningfully change most readers understand the text? Answer with Yes or No.\n"
                else:
                    out_path = os.path.join(out_dir, "predictions_zero.json")
                preds = classify(split_samples, intro, model, api_keys, out_path)

                print(f"Saved predictions to {out_path}")