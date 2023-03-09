import json

from typing import List, Dict, Tuple, Any, Union

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        examples = [json.loads(line) for line in f]
    
    return examples 

def create_resplit_index():
    original_train_examples = read_jsonl_file("data/mathqa/train-python.jsonl")
    original_dev_examples = read_jsonl_file("data/mathqa/val-python.jsonl")
    original_all_examples = original_train_examples + original_dev_examples

    text_to_example_idx = {x["text"].strip(): i for i, x in enumerate(original_all_examples)}

    dedup_train_examples = read_jsonl_file("data/mathqa/train_dedup.jsonl")
    dedup_dev_examples = read_jsonl_file("data/mathqa/val_dedup.jsonl")

    dedup_train_idx = [text_to_example_idx[x["text"].strip()] for x in dedup_train_examples]
    dedup_dev_idx = [text_to_example_idx[x["text"].strip()] for x in dedup_dev_examples]

    with open("preprocessing/mathqa_python_resplit_info.json", "w+") as f:
        info = {
            "train_first_examples": dedup_train_examples[:5],
            "dev_first_examples": dedup_dev_examples[:5],
            "train_idx": dedup_train_idx,
            "dev_idx": dedup_dev_idx
        }
        json.dump(info, f)

def recreate_split():
    # load the precomputed resplit info
    with open("preprocessing/mathqa_python_resplit_info.json", "r") as f:
        resplit_info = json.load(f)

    # load the original examples 
    original_train_examples = read_jsonl_file("data/mathqa/train-python.jsonl")
    original_dev_examples = read_jsonl_file("data/mathqa/val-python.jsonl")
    original_all_examples = original_train_examples + original_dev_examples

    # recreate the split using the resplit info
    dedup_train_examples = [original_all_examples[i] for i in resplit_info["train_idx"]]
    dedup_dev_examples = [original_all_examples[i] for i in resplit_info["dev_idx"]]

    # rename the task ids
    for i, instance in enumerate(dedup_train_examples):
        instance['task_id'] = f"train_{i}"
    for i, instance in enumerate(dedup_dev_examples):
        instance['task_id'] = f"val_{i}"

    # verify that the split is correct
    assert all([x[0]["text"] == x[1]["text"] for x in zip(dedup_train_examples[:5], resplit_info["train_first_examples"])]), "train split is incorrect"
    assert all([x[0]["text"] == x[1]["text"] for x in zip(dedup_dev_examples[:5], resplit_info["dev_first_examples"])]), "dev split is incorrect"

    # write the recreated split to disk
    with open("data/mathqa/train_dedup.jsonl", "w+") as f:
        for example in dedup_train_examples:
            f.write(json.dumps(example) + "\n")
    
    with open("data/mathqa/val_dedup.jsonl", "w+") as f:
        for example in dedup_dev_examples:
            f.write(json.dumps(example) + "\n")

def main():
    # create_resplit_index()
    recreate_split()

if __name__ == "__main__":
    main()