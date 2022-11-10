import json

from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Union
from execution.execution_evaluation import execution_acc, mathqa_execution, execution_eval_at_k
from prompting.model_prompting import codex_prompting
from prompting.prompting_utils import text_to_code_prompting, mathqa_answer_prompting

if __name__ == '__main__':
    # load the mathqa dataset with states
    mathqa_json_examples = []
    with open('data/mathqa/val_python_with_states.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            mathqa_json_examples.append(json.loads(line))

    # take the first 5 examples as few shots
    mathqa_few_shot_examples = mathqa_json_examples[:4]
    few_shot_text_list = [example['text'] for example in mathqa_few_shot_examples]
    few_shot_code_list = [example['code'] for example in mathqa_few_shot_examples]

    # FIXME: only evaluate on the first 20 examples for now
    mathqa_json_examples = mathqa_json_examples[5:105]

    def dict_avg(dict_list: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
        result_avg_dict = {}

        # first find out which items are numeric using the first example
        numeric_keys = []
        for key, value in dict_list[0].items():
            if isinstance(value, float) or isinstance(value, int):
                numeric_keys.append(key)

        for key in numeric_keys:
            sum_ = 0
            for dict_ in dict_list:
                sum_ += dict_[key]
            avg = sum_ / len(dict_list)

            result_avg_dict[key] = avg

        return result_avg_dict
        

    # dictionary result for various execution metrics
    result_dicts: List[Dict[str, Any]] = []

    for exp in tqdm(mathqa_json_examples):
        result_dict = {"example": exp}

        # first test the execution for gold solution
        exec_acc = execution_acc(exp["code"], mathqa_execution, exp["answer"])
        result_dict["gold_exec_acc"] = exec_acc[0]
        result_dict["gold_exec_rate"] = exec_acc[1]
        result_dict["gold_sol"] = exp["code"]

        def log_performance(result_dict, exec_acc, exec_rate, prompt, sol, setting_name):
            result_dict[f"{setting_name}_acc_at_k"] = exec_acc
            result_dict[f"{setting_name}_pass_at_k"] = exec_rate
            # result_dict[f"{setting_name}_exec_acc"] = exec_acc
            # result_dict[f"{setting_name}_exec_rate"] = exec_rate
            result_dict[f"{setting_name}_prompt"] = prompt
            result_dict[f"{setting_name}"] = sol


        # For writing the whole solution from NL, test the zero-shot performance
        zero_shot_prompt = text_to_code_prompting([], [], exp["text"])
        zero_shot_codex_completion = codex_prompting(prompt=zero_shot_prompt, stop="#", max_tokens=100, 
                                                     n_completions=80, temperature=0.8)
        exec_acc = execution_eval_at_k(zero_shot_codex_completion, mathqa_execution, exp["answer"], k=80)
        log_performance(result_dict, exec_acc[0], exec_acc[1], zero_shot_prompt, 
                        zero_shot_codex_completion, "sol_zero_shot")

        # For writing the whole solution from NL, test the few-shot performance
        few_shot_prompt = text_to_code_prompting(few_shot_text_list, few_shot_code_list, exp["text"])
        few_shot_codex_completion = codex_prompting(prompt=few_shot_prompt, stop="#", max_tokens=100, 
                                                    n_completions=80, temperature=0.8)
        exec_acc = execution_eval_at_k(few_shot_codex_completion, mathqa_execution, exp["answer"], k=80)
        log_performance(result_dict, exec_acc[0], exec_acc[1], few_shot_prompt, 
                        few_shot_codex_completion, "sol_few_shot")

        """
        # For only writing the answer formula given the correct context code, test the zero-shot performance
        zero_shot_prompt, context_sol_code = mathqa_answer_prompting([], exp, answer_state=False, add_gold_answer=False)
        zero_shot_codex_completion = context_sol_code + \
                                     codex_prompting(prompt=zero_shot_prompt, stop="#", max_tokens=100)[0]
        exec_acc = execution_acc(zero_shot_codex_completion, mathqa_execution, exp["answer"])
        log_performance(result_dict, exec_acc[0], exec_acc[1], zero_shot_prompt, 
                        zero_shot_codex_completion, "context_zero_shot")

        # For only writing the answer formula given the correct context code, test the zero-shot performance w/ state
        zero_shot_prompt, context_sol_code = mathqa_answer_prompting([], exp, answer_state=True, add_gold_answer=False)
        zero_shot_codex_completion = context_sol_code + \
                                     codex_prompting(prompt=zero_shot_prompt, stop="#", max_tokens=100)[0]
        exec_acc = execution_acc(zero_shot_codex_completion, mathqa_execution, exp["answer"])
        log_performance(result_dict, exec_acc[0], exec_acc[1], zero_shot_prompt, 
                        zero_shot_codex_completion, "context_state_zero_shot")

        # For only writing the answer formula given the correct context code, test the few-shot performance
        few_shot_prompt, context_sol_code = mathqa_answer_prompting(mathqa_few_shot_examples, exp, 
                                                                    answer_state=False, add_gold_answer=False)
        few_shot_codex_completion = context_sol_code + \
                                     codex_prompting(prompt=few_shot_prompt, stop="#", max_tokens=100)[0]
        exec_acc = execution_acc(few_shot_codex_completion, mathqa_execution, exp["answer"])
        log_performance(result_dict, exec_acc[0], exec_acc[1], few_shot_prompt, 
                        few_shot_codex_completion, "context_few_shot")

        # For only writing the answer formula given the correct context code, test the few-shot performance w/ state
        few_shot_prompt, context_sol_code = mathqa_answer_prompting(mathqa_few_shot_examples, exp, 
                                                                    answer_state=True, add_gold_answer=False)
        few_shot_codex_completion = context_sol_code + \
                                     codex_prompting(prompt=few_shot_prompt, stop="#", max_tokens=100)[0]
        exec_acc = execution_acc(few_shot_codex_completion, mathqa_execution, exp["answer"])
        log_performance(result_dict, exec_acc[0], exec_acc[1], few_shot_prompt, 
                        few_shot_codex_completion, "context_state_few_shot")
        """

        result_dicts.append(result_dict)

    avg_exec_acc_dict = dict_avg(result_dicts)

    print(avg_exec_acc_dict)

    with open('mathqa_zero_few_shot_full_sol_eval_at_k.json', 'w+') as f:
        json.dump(result_dicts, f, indent=2)

