import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score


def read_results(csv_file):
    df = pd.read_csv(
        csv_file,
        header=None,
        dtype=str,
        on_bad_lines="skip",   # skip malformed CSV lines
    )

    # keep only rows with exactly 4 populated columns
    df = df[df.notna().sum(axis=1) == 4].copy()

    # keep only first four columns
    df = df.iloc[:, :4]
    df.columns = ["doc_id", "correct_class", "predicted_class", "confidence"]

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Treat small overshoots and undershoots as 1.0
    df.loc[
        (df["confidence"] > 0.9999) & (df["confidence"] <= 1.0001),
        "confidence"
    ] = 1.0

    df["correct_class"] = df["correct_class"].astype(str).str.strip()
    valid_classes = set(df["correct_class"].astype(str).str.strip())
    df["predicted_class"] = df["predicted_class"].astype(str).str.strip()
    df["is_none"] = ~df["predicted_class"].astype(str).str.strip().isin(valid_classes)
    df["is_correct"] = df["correct_class"].eq(df["predicted_class"])

    return df

def plot_coverage_vs_accuracy(dfs, labels=None, output_file=None):
    def coverage_accuracy_points(df):
        df = df.copy()
        df = df.dropna(subset=["confidence"])
        df = df.sort_values("confidence", ascending=False)

        coverages = []
        accuracies = []

        for k in range(1, len(df) + 1):
            subset = df.iloc[:k]

            coverages.append(k / len(df))
            accuracies.append(subset["is_correct"].mean())

        return coverages, accuracies

    if labels is None:
        labels = [f"df{i+1}" for i in range(len(dfs))]

    plt.figure()

    for df, label in zip(dfs, labels):
        coverages, accuracies = coverage_accuracy_points(df)
        plt.plot(coverages, accuracies, label=label)

    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Coverage vs. Accuracy")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()



def plot_accuracy_vs_min_confidence(dfs, labels=None, output_file=None):
    if labels is None:
        labels = [f"df{i+1}" for i in range(len(dfs))]

    plt.figure()

    for df, label in zip(dfs, labels):
        df = df.copy()
        df = df.dropna(subset=["confidence"])

        thresholds = np.linspace(0, 1, 101)
        accuracies = []

        for threshold in thresholds:
            subset = df[df["confidence"] >= threshold]

            if len(subset) == 0:
                continue

            accuracies.append(subset["is_correct"].mean())

        plt.plot(thresholds, accuracies, label=label)

    plt.xlabel("Minimum Confidence")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Minimum Confidence")
    plt.xlim(0, 1.0)
    plt.ylim(0.4, 1.00)
    plt.grid(True)
    plt.legend()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


def get_accuracy(df):
    return accuracy_score(df["correct_class"], df["predicted_class"])
    
def get_roc_auc(df):
    return roc_auc_score(df["is_correct"].astype(int), df["confidence"])
    
def get_none_count(df):
    return int(df["is_none"].sum())

def main():
    df = read_results(sys.argv[1])
    non_none = df[~df["is_none"]].copy()

    print(f"Total rows: {len(df)}")
    print(f"Accuracy: {get_accuracy(df):.6f}")
    print(f"AUC-ROC: {get_roc_auc(df):.6f}")
    print(f'Number of "None" responses: {get_none_count(df)}')
    print(f'Accuracy without "None" responses: {get_accuracy(non_none):.6f}')

    datasets = [df]
    for filename in sys.argv[2:]:
      datasets.append(read_results(filename))
    plot_coverage_vs_accuracy(datasets, sys.argv[1:], output_file='cov-acc.png')
    plot_accuracy_vs_min_confidence(datasets, sys.argv[1:], output_file='conf-acc.png')
    
if __name__ == "__main__":
    main()

