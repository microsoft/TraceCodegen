import json

from itertools import chain
from typing import List, Dict, Any

PRED_FILES = [f"amlt/mathqa-dedup-mle-pass_80_eval_train-bs_1/gpt-neo-mathqa-finetuning/" \
                f"lightning_logs/version_0/predictions_step_1_rank_{i}.jsonl" for i in range(16)]

SAVE_FILE = "mathqa_dedup_train_samples.jsonl"

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def main():
    # read the files
    examples = list(chain(*[read_jsonl_file(file_path) for file_path in PRED_FILES]))

    task_id_set = set()
    unique_program_pct = []
    with open(SAVE_FILE, "w") as f:
        for example in examples:
            task_id = example["metadata"]["task_id"]
            if task_id in task_id_set:
                continue
            else:
                task_id_set.add(task_id)

            # calculate some statistics by the way
            example["generated_k_programs"]
            unique_program_pct.append(len(set(example["generated_k_programs"])) / len(example["generated_k_programs"]))

            f.write(json.dumps(example) + "\n")

    print(f"There are {len(task_id_set)} unique tasks")
    print(f"The average number of unique programs is {sum(unique_program_pct) / len(unique_program_pct)}")

if __name__ == "__main__":
    main()