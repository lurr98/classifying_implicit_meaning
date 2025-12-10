import argparse, json, requests, torch, spacy, pickle, os
import numpy as np
from prepare_data import read_json
from openai import OpenAI
from collections import Counter
from datasets import Dataset
from prepare_data import format_few_examples, prepare_linear_classifier
from transformers import AutoTokenizer, AutoModelForCausalLM
# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from scipy.sparse import hstack, csr_matrix
from sklearn.tree import DecisionTreeClassifier
from typing import Union
# feature importances
from interpret_models import * 

prompt = "You are an annotator that has to annotate the data based on the following instruction:\n\n<break>\n\nAnswer with the label and in the new line the reasoning behind your decision. Example: 'Label\nExplain your decision in this new line.' Keep the explanation short, 10 words maximum.\n\n"


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

def classify_samples_gpt(item_dict: dict, api_file: str, instructions: str, model: str) -> dict:

    with open(api_file, "r") as af:
        api_keys = af.readlines()

    api_key = api_keys[0].split(' ')[-1].strip()
    print(api_key)

    predictions = {}

    # client = OpenAI(api_key=api_key)

    for i, (sample_id, item) in enumerate(item_dict.items()):
        formatted_item = f"{item['article_name']}\nFirst Text:\n{item['context_before']} **{item['sentence_1']}** {item['context_after']}\n\nSecond Text:\n{item['context_before']} **{item['sentence_2']}** {item['context_after']}"

        # TODO: make sure this works
        instructions_final = instructions.join(prompt.split("break"))

        if i == 0:
            print(instructions_final)
        print(formatted_item)
        print("-----")

        # messages = [ {"role": "system", "content": instructions_final}, {"role": "user", "content": formatted_item}]
        # chat = client.chat.completions.create(model=model, messages=messages)
        # reply = chat.choices[0].message.content
        # 
        # predictions[sample_id] = reply
# 
        # if i % 10 == 0:
        #     print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    return predictions


def classify_samples_deepseek(item_dict: dict, api_file: str, instructions: str) -> dict:

    with open(api_file, "r") as af:
        api_keys = af.readlines()

    api_key = api_keys[1].split(" ")[-1].strip()

    predictions = {}
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for i, (sample_id, item) in enumerate(item_dict.items()):
        formatted_item = f"{item['article_name']}\nFirst Text:\n{item['context_before']} **{item['sentence_1']}** {item['context_after']}\n\nSecond Text:\n{item['context_before']} **{item['sentence_2']}** {item['context_after']}"

        instructions_final = instructions.join(prompt.split("break"))

        messages = [{"role": "system", "content": instructions_final}, {"role": "user", "content": formatted_item}]
        chat = client.chat.completions.create(model="deepseek-chat", messages=messages)
        reply = chat.choices[0].message.content
        predictions[sample_id] = reply

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


def classify_samples_huggingface(item_dict: dict, instructions: str, model: str) -> dict:

    tokenizer = AutoTokenizer.from_pretrained(model, token="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")
    model = AutoModelForCausalLM.from_pretrained(model, token="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")


    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    predictions = {}

    for i, (sample_id, item) in enumerate(item_dict.items()):
        formatted_item = f"{item['article_name']}\nFirst Text:\n{item['context_before']} **{item['sentence_1']}** {item['context_after']}\n\nSecond Text:\n{item['context_before']} **{item['sentence_2']}** {item['context_after']}"

        instructions_final = instructions.join(prompt.split("break"))

        prompt = instructions_final + formatted_item
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)

        # Slice off the prompt tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]

        # Decode ONLY the new tokens
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        predictions[sample_id] = reply.strip()

        if i % 10 == 0:
            print('Item {} from {}'.format(str(i), len(item_dict.keys())))
    
    return predictions


## simple ML models

def pos_tagger(texts: list) -> list:

    nlp = spacy.load("en_core_web_lg") 
    pos_tags = []
    for text in texts:
        doc = nlp(text)
        pos_tags.append(" ".join([t.pos_ for t in doc]))

    return pos_tags


def linear_classification(vector: list, targets: list, ids: list, mode: str, classifier: str, out: str) -> None:

    preds_final = {}
    # if mode == "pos" or mode == "both":
    #     pos_vectorizer = TfidfVectorizer(ngram_range=(2,3), token_pattern=r"[^ ]+", min_df=2)
    #     pos_features = pos_vectorizer.fit_transform(pos_tagger(texts))
    # if mode == "lex" or mode == "both":
    #     lex_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    #     lex_features = lex_vectorizer.fit_transform(texts)
    # 
    # if mode == "both":
    #     vector = hstack([pos_features, lex_features])
    # elif mode == "pos":
    #     vector = pos_features
    # elif mode == "lex":
    #     vector = lex_features

    if classifier == "logistic":
        clf = LogisticRegression()
    elif classifier == "svm":
        clf = SVC()
    elif classifier == "tree":
        clf = DecisionTreeClassifier()

    preds = cross_val_predict(clf, vector, targets, cv=LeaveOneOut())

    for i, pred in enumerate(preds):
        preds_final[ids[i]] = pred

    return preds_final


def train_DT(train_data: list | np.ndarray, targets: list, out_name: str, grid: bool) -> DecisionTreeClassifier:
    # train an DT classifier using scikit-learn

    if grid:
        print('Using grid search to find the best parameter combination.')
        print('Paramter options: {"max_depth":[30, 50], "max_features":("sqrt", "log2", None)}')
        parameters = {'max_depth':[30, 50], 'max_features':('sqrt', 'log2', None)}
        dt = DecisionTreeClassifier()
        dt_clf = GridSearchCV(dt, parameters, scoring='accuracy', n_jobs=-1, verbose=2)
        dt_clf.fit(train_data, targets)
    else:
        # default hyperparameters: {max_depth=50, max_features=None}
        dt_clf = DecisionTreeClassifier(max_depth=50, max_features=None)
        dt_clf.fit(train_data, targets)

    save_model(dt_clf, f"linear_models/DT_{out_name}")

    return dt_clf


def save_model(model: Union[DecisionTreeClassifier, GridSearchCV], filename: str) -> None:

    # save the model to disk
    pickle.dump(model, open(filename, 'wb')) 


#################################
#        Linear models          #
#################################

def load_linear_model(model_path: str) -> DecisionTreeClassifier:

    # load classifier
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    return classifier


def predict_labels(classifier: DecisionTreeClassifier, features: np.ndarray) -> list:
    # let the classifier predict the labels and return the predictions

    predictions = classifier.predict(features)

    return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('sample_dir', type=str, help='name of the folder containing the relevant samples')
    parser.add_argument('api', type=str, help='name of the file containing API key')
    parser.add_argument('model', type=str, help='name of the model to be used')
    parser.add_argument('outfile', type=str, help='name of the output file containing the predictions')
    parser.add_argument('-i', '--intro', nargs='?', help='name of the introduction file to be used')
    parser.add_argument('-f', '--few_shot', nargs='?', help='number of few shot examples that will be presented to the model')
    parser.add_argument('-l', '--linear', nargs='?', help='mode of linear classification ("pos", "lex" or "both")')
    parser.add_argument('-ling', '--ling_features', action='store_true')
    parser.add_argument('-z', '--zero_shot', action='store_true')
    parser.add_argument('-t', '--tie', action='store_true')
    parser.add_argument('-mj', '--majority_bs', action='store_true')
    parser.add_argument('-adj', '--adjective_bs', action='store_true')

    args = parser.parse_args()

    # if args.intro:
    #     intros = "instructions/" + args.intro
    # else:
    #     intros_zf = "instructions/instructions_zero" if args.zero_shot else "instructions/instructions_few"
    #     intros = intros_zf + "_tie.md" if args.tie else intros_zf + ".md"  

    print(f"Results will be saved in {args.outfile}")
    notie = "" if args.tie else "_noties"

    for filename in os.listdir(args.sample_dir):
        topic = filename[:-13]
        samples = read_json(args.sample_dir + filename)

        if args.few_shot:
            few = args.few_shot
            few_shot_samples = f"few_shot_samples/few_shots_merged{notie}_{topic}.json"
            intro = format_few_examples(few_shot_samples, "samples_merged.json", f"instructions/instructions_few{notie}.md", few)
        elif args.zero_shot:
            with open(f"instructions/instructions_zero{notie}.md", "r") as i:
                intro = i.read()

        if args.majority_bs:
            model_name = "baselines"
            preds = majority_baseline(samples, "No")
        elif args.adjective_bs:
            model_name = "baselines"
            preds = adj_baseline(samples, "Yes", "No") 
        elif args.linear:
            if len(args.samples.split(",")) == 2:
                first_train_test, first_feature_names = prepare_linear_classifier(args.samples.split(","), args.linear, args.ling_features)
                sec_train_test, sec_feature_names = prepare_linear_classifier(args.samples.split(",")[::-1], args.linear, args.ling_features)
                first_dt = train_DT(first_train_test[0][0], first_train_test[0][1], args.samples.split(",")[0].split("/")[-1], True)
                plot_feature_importance(get_coefficients(first_dt.best_estimator_, "dt", first_feature_names), f"dt_model_{args.samples.split(',')[0].split('/')[-1]}")
                sec_dt = train_DT(sec_train_test[0][0], sec_train_test[0][1], args.samples.split(",")[1].split("/")[-1], True)
                plot_feature_importance(get_coefficients(sec_dt.best_estimator_, "dt", sec_feature_names), f"dt_model_{args.samples.split(',')[1].split('/')[-1]}")
                first_preds = {first_train_test[1][2][i]: pred for i, pred in enumerate(predict_labels(first_dt, first_train_test[1][0]))}
                sec_preds = {sec_train_test[1][2][i]: pred for i, pred in enumerate(predict_labels(sec_dt, sec_train_test[1][0]))}
                preds = {**first_preds, **sec_preds}
            else:
                vectors, targets, ids = prepare_linear_classifier(args.samples.split(","), args.linear)
                preds = linear_classification(vectors, targets, ids, args.linear, args.model, args.outfile)
            model_name = "linear"
        elif "gpt" in args.model:
            model_name = "GPT"
            preds = classify_samples_gpt(samples, args.api, intro, args.model)
        elif args.model == "deepseek":
            model_name = args.model
            preds = classify_samples_deepseek(samples, args.api, intro)
        elif args.model == "mv":
            preds = majority_baseline(samples, "No")
        elif args.model == "adj":
            preds = adj_baseline(samples, "Yes", "No")
        else:
            model_name = "mistral" if "mistral" in args.model else "llama" 
            preds = classify_samples_huggingface(args.samples, intro, args.model)

        exit()

        if args.linear:
            exp_str = "leave_one_out"
        else:
            exp_str = "200_held_out" if len(list(samples.keys())) == 200 else "108_cross_validation"

        with open(f"predictions/classification/{exp_str}/{model_name}/" + args.outfile, "w") as gpt:
            json.dump(preds, gpt)
