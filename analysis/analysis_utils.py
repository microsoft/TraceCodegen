import json

from typing import List, Dict, Any

train_file_path = "data/mathqa/train-python.jsonl"
val_file_path = "data/mathqa/val-python.jsonl"
test_file_path = "data/mathqa/test-python.jsonl"

def load_examples(file_path: str) -> List[Dict[str, Any]]:
    """
    Load examples from jsonl file.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_examples_from_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load examples from jsonl file.
    """
    result = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            result.extend([json.loads(line) for line in f])

    return result

def binning(bins: List[int], values: List[int], title: str = "") -> None:
    """
    Binning values into bins.
    """
    bins = sorted(bins)
    binned_values = [[] for _ in range(len(bins)+1)]

    for val in values:
        for i, bin in enumerate(bins):
            if val <= bin:
                binned_values[i].append(val)
                break
        else:
            binned_values[-1].append(val)

    # change the length to percentages
    binned_percentage = [len(binned_values[i]) * 100 / len(values) for i in range(len(bins)+1)]

    # print the histogram
    print("----------------------------------------------------")
    print("                  {}".format(title))
    print("----------------------------------------------------")
    for i, percentage in enumerate(binned_percentage):
        if i != len(binned_percentage) - 1:
            line_str = f"x<={bins[i]}:\t"
        else:
            line_str = f"x>{bins[i-1]}:\t"

        line_str += "|" * int(percentage) + f" {percentage:.2f}%"
        print(line_str)
    print("----------------------------------------------------")


