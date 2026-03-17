import json, argparse
import numpy as np
from scipy import stats

selected_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET"]

# --- 1. Welch's t-test ---
def welchs_t_test(first_label: np.array, second_label: np.array):
    t_stat, p_t = stats.ttest_ind(first_label, second_label, equal_var=False)
    print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
    return t_stat, p_t

# --- 2. Mann–Whitney U test ---
def mann_whitney_u_test(first_label: np.array, second_label: np.array):
    u_stat, p_u = stats.mannwhitneyu(first_label, second_label, alternative='two-sided')
    print(f"Mann–Whitney U: U = {u_stat}, p = {p_u:.4f}")
    return u_stat, p_u

# --- 3. Bootstrap CI for mean difference ---
def bootstrap_mean_diff(first_label: np.array, second_label: np.array, n_bootstrap=10000, ci=95):
    def mean_diff(x, y):
        return np.mean(x) - np.mean(y)

    # scipy.stats.bootstrap expects data as a tuple of arrays
    res = stats.bootstrap(
        data=(first_label, second_label),
        statistic=mean_diff,
        paired=False,        # samples are independent
        vectorized=False,    # our function isn't vectorized
        n_resamples=10000,   # number of bootstrap draws
        confidence_level=0.95,
        method="percentile", # or "basic", "BCa" if available in your SciPy version
        random_state=42
    )

    return res.confidence_interval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tba')
    parser.add_argument('analysis', type=str, help='name of the file containing the linguistic analysis')
    parser.add_argument('-m', '--metrics', type=str, help='metrics of interest, separated by comma')
    parser.add_argument('output', type=str, help='name of the output file')

    args = parser.parse_args()

    with open(args.analysis) as jsn:
        analysis = json.load(jsn)

    significance_dict = {}

    # TODO: maybe simply change analysis to dict with IDs as keys
    labels = sorted(list(set(analysis["targets"])))
    class_indices = [[i for i, ids in enumerate(analysis["IDs"]) if analysis["targets"][i] == curr_class] for curr_class in labels]
    for metric in args.metrics.split(","):
        significance_dict[metric] = {}
        if metric == "POS_distr":
            for sel_pos in selected_POS:
                significance_dict[metric][sel_pos] = {}
        for i, label in enumerate(class_indices):
            for j, other_label in enumerate(class_indices):
                if i != j:
                    if type(analysis[metric][0]) in [int, float]:
                        first_values, second_values = [analysis[metric][idx] for idx in label], [analysis[metric][idx] for idx in other_label]
                        t_test = welchs_t_test(np.array(first_values), np.array(second_values))
                        u_test = mann_whitney_u_test(np.array(first_values), np.array(second_values))
                        confidence_interval = bootstrap_mean_diff(np.array(first_values), np.array(second_values))
    
                        if not f"({labels[j]}, {labels[i]})" in significance_dict[metric]:
                            significance_dict[metric][f"({labels[i]}, {labels[j]})"] = {"T Test": t_test, "U Test": u_test, "Bootstrap": confidence_interval}
                            # significance_dict[metric][f"({labels[i]}, {labels[j]})"] = {"T Test": t_test, "U Test": u_test}

                    elif metric == "POS_distr":
                        for pos in selected_POS:
                            first_values, second_values = [], []
                            for first_idx in label:
                                try:
                                    first_values.append(analysis[metric][first_idx][pos])
                                except KeyError:
                                    first_values.append(0)

                            for second_idx in other_label:
                                try:
                                    second_values.append(analysis[metric][second_idx][pos])
                                except KeyError:
                                    second_values.append(0)

                            t_test = welchs_t_test(np.array(first_values), np.array(second_values))
                            u_test = mann_whitney_u_test(np.array(first_values), np.array(second_values))
                            confidence_interval = bootstrap_mean_diff(np.array(first_values), np.array(second_values))
    
                            if not f"({labels[j]}, {labels[i]})" in significance_dict[metric][pos]:
                                significance_dict[metric][pos][f"({labels[i]}, {labels[j]})"] = {"T Test": t_test, "U Test": u_test, "Bootstrap": confidence_interval}
                                # significance_dict[metric][pos][f"({labels[i]}, {labels[j]})"] = {"T Test": t_test, "U Test": u_test}


    with open(args.output, "w") as jsn:
        json.dump(significance_dict, jsn)