from typing import List, Tuple, Dict

from analysis.analysis_utils import load_examples_from_files
from analysis.mathqa_train_test_overlap import get_overlap_example_ids

result_files_state_mask_non_val_no_skip = [f"amlt/mathqa-125M-state-mask-non-val/" + \
                f"gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/predictions_step_48543_rank_{i}.jsonl" for i in range(16)]
result_files_state_mask_non_val = [f"amlt/mathqa-125M-state-mask-non-val-skip-ts/" + \
                f"gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/predictions_step_53279_rank_{i}.jsonl" for i in range(16)]
result_files_state = [f"amlt/mathqa-125M-state-skip-ts/" + \
                f"gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/predictions_step_49727_rank_{i}.jsonl" for i in range(16)]
result_files_baseline = [f"amlt/mathqa-finetune-gpt-neo-125M-pad-left/" + \
                f"gpt-neo-mathqa-finetuning/lightning_logs/version_0/predictions_step_54044_rank_{i}.jsonl" for i in range(16)]

def filtered_results(result_files: List[str], min_dist: int) -> None:
    examples = load_examples_from_files(result_files)
    overlapping_example_ids = get_overlap_example_ids(set_name="val", min_allow_dist=min_dist)

    filtered_exec_acc = list(map(lambda x: x["metrics"]["exec_acc"], filter(lambda x: not x["metadata"]["task_id"] in overlapping_example_ids, examples)))
    print(f"originally {len(examples)} examples, after filtering {len(filtered_exec_acc)} examples")

    print(f"original exec_acc: {sum(map(lambda x: x['metrics']['exec_acc'], examples)) / len(examples)}")
    print(f"after filtering exec_acc: {sum(filtered_exec_acc) / len(filtered_exec_acc)}")

if __name__ == "__main__":
    filtered_results(result_files_baseline, min_dist=2)
    filtered_results(result_files_state_mask_non_val, min_dist=2)