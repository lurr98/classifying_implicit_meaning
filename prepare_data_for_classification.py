import argparse, os
from classification.prepare_data import read_json, filter_ties, prepare_few_shot, separate_test_data, format_few_examples


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare data for classification")
    parser.add_argument("input_dir", type=str, help="Input directory containing relevant study data")
    parser.add_argument("output_dir", type=str, help="Output directory for prepared data")
    args = parser.parse_args()


    for subdir in os.listdir(args.input_gold):
        topic = subdir[:-6]
        samples = read_json(os.path.join(subdir, "current_samples.json"))
        samples_noties = {idx: sample for idx, sample in samples.items() if sample["ID"] in gold_noties}

        with open(f"{topic}_study/samples/samples.json", "w") as jsn:
            json.dump(samples, jsn)

        with open(f"{topic}_study/samples/samples_noties.json", "w") as jsn:
            json.dump(samples_noties, jsn)

        gold = read_json(os.path.join(subdir, "gold_standard", "gold_standard.json"))
        gold_noties = filter_ties(gold)

        with open(f"{topic}_study/gold_standard/gold_standard.json", "w") as jsn:
            json.dump(gold, jsn)

        with open(f"{topic}_study/gold_standard/gold_standard_noties.json", "w") as jsn:
            json.dump(gold_noties, jsn)

        few_shots_organized, few_shots_collection = prepare_few_shot(gold, [2, 4, 8, 16], ["No", "Yes", "Tie"], ["No", "Yes"])
        few_shots_noties_organized, few_shots_noties_collection = prepare_few_shot(gold_noties, [2, 4, 8, 16], ["No", "Yes"], ["No", "Yes"])

        with open(f"{topic}_study/few_shots/few_shots_organized.json", "w") as jsn:
            json.dump(few_shots_organized, jsn)

        with open(f"{topic}_study/few_shots/few_shots_collection.json", "w") as jsn:
            json.dump(few_shots_collection, jsn)

        with open(f"{topic}_study/few_shots/few_shots_noties_organized.json", "w") as jsn:
            json.dump(few_shots_noties_organized, jsn)

        with open(f"{topic}_study/few_shots/few_shots_noties_collection.json", "w") as jsn:
            json.dump(few_shots_noties_collection, jsn)

        test_data = separate_test_data(samples, few_shots_collection)
        test_data_noties = separate_test_data(samples_noties, few_shots_noties_collection)

        with open(f"{topic}_study/test_data/test_samples.json", "w") as jsn:
            json.dump(test_data, jsn)

        with open(f"{topic}_study/test_data/test_samples_noties.json", "w") as jsn:
            json.dump(test_data_noties, jsn)

        