#!/usr/bin/env python3

import argparse
import csv
import sys


def choose_hybrid_result(t1, t2):
    """
    Given:
        t1 = (predicted_class_1, normalized_confidence_1)
        t2 = (predicted_class_2, normalized_confidence_2)

    Returns:
        (chosen_class, chosen_confidence)
    """
    class1, conf1 = t1
    class2, conf2 = t2

    if class1 == class2:
        return class1, 1.0 - ((1.0 - conf1)*(1.0 - conf2))

    if conf1 >= conf2:
        return class1, conf1*(1.0-conf2)
    else:
        return class2, conf2*(1.0-conf1)


def read_results(filename):
    results = {}
    confidences = []

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) != 4:
                print(f"Skipping malformed row in {filename}: {row}", file=sys.stderr)
                continue

            doc_id, correct_class, predicted_class, confidence = row

            try:
                confidence = float(confidence)
            except ValueError:
                print(f"Skipping row with bad confidence in {filename}: {row}", file=sys.stderr)
                continue

            results[doc_id] = {
                "correct_class": correct_class,
                "predicted_class": predicted_class,
                "raw_confidence": confidence,
            }

            confidences.append(confidence)

    return results, confidences


def normalize_results(results, confidences):
    if not confidences:
        return

    min_conf = min(confidences)
    max_conf = max(confidences)
    print('min confidence:',min_conf, file=sys.stderr)
    print('max confidence:',max_conf, file=sys.stderr)

    if max_conf == min_conf:
        # Degenerate case: every confidence identical
        for doc_id in results:
            results[doc_id]["confidence"] = 1.0
        return

    for doc_id in results:
        raw = results[doc_id]["raw_confidence"]
        norm = (raw - min_conf) / (max_conf - min_conf)
        results[doc_id]["confidence"] = norm


def main():
    parser = argparse.ArgumentParser(
        description="Combine two document-classification result CSVs into a hybrid classifier."
    )
    parser.add_argument("file1", help="CSV results from technique 1")
    parser.add_argument("file2", help="CSV results from technique 2")
    parser.add_argument(
        "-o", "--output",
        default="-",
        help="Output CSV file, or '-' for stdout"
    )

    args = parser.parse_args()

    results1, confs1 = read_results(args.file1)
    results2, confs2 = read_results(args.file2)

    print('normalizing confidences from classifer 1:', file=sys.stderr)
    normalize_results(results1, confs1)
    print('normalizing confidences from classifer 2:', file=sys.stderr)
    normalize_results(results2, confs2)

    common_doc_ids = sorted(set(results1) & set(results2))

    missing_from_1 = set(results2) - set(results1)
    missing_from_2 = set(results1) - set(results2)

    if missing_from_1:
        print(f"Warning: {len(missing_from_1)} docs missing from file1", file=sys.stderr)

    if missing_from_2:
        print(f"Warning: {len(missing_from_2)} docs missing from file2", file=sys.stderr)

    out_f = sys.stdout if args.output == "-" else open(
        args.output, "w", newline="", encoding="utf-8"
    )
    
    with out_f:
        writer = csv.writer(out_f)
        correct = 0
        correct_is_possible = 0
        for doc_id in common_doc_ids:
            r1 = results1[doc_id]
            r2 = results2[doc_id]

            if r1["correct_class"] != r2["correct_class"]:
                print(
                    f"Warning: correct class mismatch for {doc_id}: "
                    f"{r1['correct_class']} vs {r2['correct_class']}",
                    file=sys.stderr
                )

            hybrid_class, hybrid_confidence = choose_hybrid_result(
                (r1["predicted_class"], r1["confidence"]),
                (r2["predicted_class"], r2["confidence"]),
            )

            if hybrid_class == r1["correct_class"]:
                correct += 1

            if r1["predicted_class"] == r1["correct_class"] or \
               r2["predicted_class"] == r2["correct_class"]: 
              correct_is_possible += 1

            writer.writerow([
                doc_id,
                r1["correct_class"],
                hybrid_class,
                hybrid_confidence,
            ])
        print('accuracy:',correct,'out of',
              len(common_doc_ids),':',(correct/len(common_doc_ids)) ,
              file=sys.stderr)
        print('maximum possible accuracy:',correct_is_possible,'out of',
              len(common_doc_ids),':',(correct_is_possible/len(common_doc_ids)) ,
              file=sys.stderr)


if __name__ == "__main__":
    main()
