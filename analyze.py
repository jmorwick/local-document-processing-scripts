import sys
import pandas as pd
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


if __name__ == "__main__":
    main()

