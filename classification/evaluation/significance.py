import json
from statsmodels.stats.contingency_tables import mcnemar

def load_json(path):
    with open(path) as f:
        return json.load(f)

def mcnemar_test(preds_a, preds_b, gold):
    # contingency table
    #            Model B
    #           correct  wrong
    # Model A correct   n11     n10
    #         wrong     n01     n00

    n11 = n10 = n01 = n00 = 0

    for k in gold:
        a_correct = preds_a[k] == gold[k]
        b_correct = preds_b[k] == gold[k]

        if a_correct and b_correct:
            n11 += 1
        elif a_correct and not b_correct:
            n10 += 1
        elif not a_correct and b_correct:
            n01 += 1
        else:
            n00 += 1

    table = [[n11, n10],
             [n01, n00]]

    result = mcnemar(table, exact=True)
    return table, result

if __name__ == "__main__":
    preds_a = load_json("preds_model_a.json")
    preds_b = load_json("preds_model_b.json")
    gold = load_json("gold.json")

    table, result = mcnemar_test(preds_a, preds_b, gold)

    print("Contingency table:")
    print(table)
    print()
    print(f"McNemar statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")

    if result.pvalue < 0.05:
        print("✅ Difference is statistically significant (p < 0.05)")
    else:
        print("❌ Difference is NOT statistically significant (p ≥ 0.05)")
