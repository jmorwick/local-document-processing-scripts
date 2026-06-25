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

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)

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

def plot_coverage_vs_accuracy(dfs, labels=None, styles=None, output_file=None):
    def coverage_accuracy_points(df):
        df = df.copy()
        df = df.sort_values("confidence", ascending=False)

        coverages = []
        accuracies = []

        for t in range(0, 100):
            filtered = df[df["confidence"] >= t/100]
            coverages.append(100*len(filtered) / len(df))
            accuracies.append(100*filtered["is_correct"].mean())
        filtered = df[df["confidence"] >= 1]
        coverages.append(0)
        accuracies.append(100*filtered["is_correct"].mean())

        return coverages, accuracies
    if labels is None:
        labels = [f"df{i+1}" for i in range(len(dfs))]

    plt.figure()

    for df, label,style in zip(dfs, labels,styles):
        coverages, accuracies = coverage_accuracy_points(df)
        print(coverages, '\n',accuracies,'\n***\n\n')
        plt.step(coverages, accuracies, label=label,linestyle=style)

    plt.xlabel("% Coverage")
    plt.ylabel("% Accuracy")
    plt.title("Coverage vs. Accuracy")
    plt.xlim(0, 100)
    plt.ylim(60, 101)
    plt.grid(True)
    plt.legend()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()



def get_accuracy(df):
    return accuracy_score(df["correct_class"], df["predicted_class"])

def get_non_none_accuracy(df):
    non_none = df[~df["is_none"]].copy()
    return accuracy_score(non_none["correct_class"], non_none["predicted_class"])
    
def get_roc_auc(df):
    return roc_auc_score(df["is_correct"].astype(int), df["confidence"])
    
def get_none_count(df):
    return int(df["is_none"].sum())

def main():
    files = []
    names = []
    styles = []
    for arg in sys.argv[1:]:
       if ',' in arg:
         tokens = arg.split(',')
         files.append(tokens[0])
         names.append(tokens[1])
         styles.append(tokens[2])
       else:
         files.append(arg)
         names.append(arg)
         styles.append('-')
    df = read_results(files[0])

    print(f"Total rows: {len(df)}")
    print(f"Accuracy: {get_accuracy(df):.6f}")
    print(f"AUC-ROC: {get_roc_auc(df):.6f}")
    print(f'Number of "None" responses: {get_none_count(df)}')
    print(f'Accuracy without "None" responses: {get_non_none_accuracy(df):.6f}')

    datasets = [df]
    for filename in files[1:]:
      datasets.append(read_results(filename))
    plot_coverage_vs_accuracy(datasets, names, styles, output_file='cov-acc.png')
    
if __name__ == "__main__":
    main()

