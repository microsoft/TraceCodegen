import json

from transformers.file_utils import PT_SPEECH_SEQ_CLASS_SAMPLE

from execution.execution_evaluation import execution_acc, mathqa_execution
from execution.execution_evaluation import execution_eval_at_k, batch_execution_acc

prediction_files = ["amlt/mathqa-gpt-j-6B-eval/gpt-neo-mathqa-finetuning/" + \
                    f"lightning_logs/version_0/predictions_step_1_rank_{i}.jsonl" for i in range(16)]

PASS_AT_K = 80

if __name__ == '__main__':
    # evaluate the performance from the prediction file(s)
    predictions = []
    for f_path in prediction_files:
        predictions += [json.loads(line) for line in open(f_path)]

    # predictions = predictions[:10] # FIXME: debug setting only

    all_generated_k_programs = [p["generated_k_programs"] for p in predictions]
    all_generated_k_programs_faltten = [item for sublist in all_generated_k_programs for item in sublist]
    gold_answers = [p["metadata"]["answer"] for p in predictions]

    result_list = batch_execution_acc(all_generated_k_programs_faltten, mathqa_execution, gold_answers, 
                                    len(predictions), PASS_AT_K)

    acc_at_k, pass_at_k = zip(*result_list)

    print("acc_at_k: ", sum(acc_at_k) / len(acc_at_k))
    print("pass_at_k: ", sum(pass_at_k) / len(pass_at_k))

    