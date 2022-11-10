import json

from tqdm import tqdm
from typing import Dict, List, Any
from bashplotlib.histogram import plot_hist
from itertools import chain
from concurrent.futures import ProcessPoolExecutor as Pool

from lightning_modules.models.gpt_util import get_gpt
from execution.execution_evaluation import mathqa_execution
from analysis.analysis_utils import binning

result_files_state = [f"amlt/mathqa-state-finetuning-125M-line-state-mask-ctx-len-1648-fixed/" + \
                f"gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/predictions_step_16575_rank_{i}.jsonl" for i in range(16)]
result_files_baseline = [f"amlt/mathqa-finetune-gpt-neo-125M-pad-left/" + \
                f"gpt-neo-mathqa-finetuning/lightning_logs/version_0/predictions_step_40833_rank_{i}.jsonl" for i in range(16)]
result_files_6B = [f"amlt/mathqa-gpt-j-6B-eval-pad-left-start_temp_0.8-max_len_256/" + \
                f"gpt-neo-mathqa-finetuning/lightning_logs/version_0/predictions_step_13_rank_{i}.jsonl" for i in range(16)]

def answer_in_code(result_dicts: List[Dict[str, Any]]):
    # analysis whether the final line about the answer is generated

    answer_in_count = 0
    for result_dict in result_dicts:
        if "answer" in result_dict["generated_program"]:
            answer_in_count += 1

    print(f"answer in code: {answer_in_count}/{len(result_dicts)}, percentage: {answer_in_count/len(result_dicts)}")

def code_line_num(result_dicts: List[Dict[str, Any]]):
    # analysis the number of code lines

    code_line_num = []
    for result_dict in result_dicts:
        line_num = len(list(filter(lambda x: not x.startswith("#") and not len(x.strip()) == 0, 
                                                result_dict["generated_program"].split("\n"))))
        # if line_num > 20:
        #     print("##########################")
        #     print(result_dict["generated_program"])
        #     print("##########################")

        code_line_num.append(line_num)

    plot_hist(code_line_num, bincount=10, binwidth=0.5, xlab=True)

def program_length_check(result_dicts: List[Dict[str, Any]]):
    _, tokenizer = get_gpt('EleutherAI/gpt-neo-125M', tokenizer_only=True)

    whole_program_length = []
    line_length = []
    program_lines = []
    for example in tqdm(result_dicts):
        code = example['code']
        whole_program_length.append(len(tokenizer.encode(code)))
        code_lines = code.split('\n')
        program_lines.append(len(code_lines))

        for line in code_lines:
            line_length.append(len(tokenizer.encode(line)))

    print(f"program length > 100 {len(list(filter(lambda x: x>100, whole_program_length)))} " + \
          f"> 128 {len(list(filter(lambda x: x>128, whole_program_length)))} " + \
          f"> 150 {len(list(filter(lambda x: x>150, whole_program_length)))} " + \
          f"> 200 {len(list(filter(lambda x: x>200, whole_program_length)))} " + \
          f"of total {len(whole_program_length)}")
    print(f"line length > 20 {len(list(filter(lambda x: x>20, line_length)))} of total {len(line_length)}")
    print(f"program lines > 20 {len(list(filter(lambda x: x>20, program_lines)))} of total {len(program_lines)}")
    print("Done")

def pass_at_k_analysis(result_dicts: List[Dict[str, Any]]):
    # gather all the top-k programs
    top_k_lists = list(map(lambda x: list(x["generated_k_programs"]), result_dicts))
    gold_answer = list(map(lambda x: x["metadata"]["answer"], result_dicts))

    # dedup the same programs in the sert of k programs for each example
    unique_top_k_lists = list(map(lambda x: list(set(x)), top_k_lists))
    flatten_unique_programs = list(chain.from_iterable(unique_top_k_lists))

    with Pool(100) as p:
        unique_executed_answers = p.map(mathqa_execution, flatten_unique_programs)
    unique_executed_answers = list(unique_executed_answers)

    k = 0
    execution_result_list = []
    for i, top_k_list in enumerate(unique_top_k_lists):
        execution_result_dict = {"unique_programs": [], "metadata": result_dicts[i]["metadata"]}
        for j, program in enumerate(top_k_list):
            match = 1.0 if unique_executed_answers[k] == gold_answer[i] else 0.0
            execution_result_dict["unique_programs"].append({"program": program, "exec_result": str(unique_executed_answers[k]), "match": match})
            k += 1
        execution_result_list.append(execution_result_dict)

    # analysis the number of unique programs 
    unique_programs_list = list(map(lambda x: len(x["unique_programs"]), execution_result_list))
    binning(bins=[2, 4, 8, 15, 20, 25, 30], values=unique_programs_list, title="unique programs per task")

    # analysis of number of unique correct programs
    unique_correct_programs_list = list(map(lambda x: len(list(filter(lambda y: y["match"] == 1.0, x["unique_programs"]))), execution_result_list))
    binning(bins=[0, 1, 2, 4, 8], values=unique_correct_programs_list, title="unique correct programs per task")

    # how many pairs of programs yields the same answer (though may not correct)
    same_answer_pairs_count_list = list(map(lambda x: len(x["unique_programs"]) - len(set(map(lambda y: y["exec_result"], x["unique_programs"]))), execution_result_list))
    binning(bins=[0, 1, 2, 3, 4, 5], values=same_answer_pairs_count_list, title="pairs of programs with same answer")


    with open("analysis/pass_at_k_analysis_6B_high_temp.json", "w") as f:
        json.dump(execution_result_list, f, indent=4)


if __name__ == "__main__":
    # load the result files
    result_dicts_baseline = []
    for result_file in result_files_6B:
        with open(result_file, "r") as f:
            for line in f:
                result_dicts_baseline.append(json.loads(line))

    # result_dicts_state = []
    # for result_file in result_files_state:
    #     with open(result_file, "r") as f:
    #         for line in f:
    #             result_dicts_state.append(json.loads(line))

    analysis_results = pass_at_k_analysis(result_dicts_baseline)
    
    print("")



# print("\n".join(list(filter(lambda x: not x.startswith("#"), failure_cases[2]["generated_program"].split("\n")))))