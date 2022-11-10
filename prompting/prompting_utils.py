from typing import List, Dict, Any, Tuple

# define some natural language descriptions of the task
task_description = "\"\"\"\n The task is to understand a question "\
                    "described in natural language and write code to get the solution\n\"\"\""
nl_intro = "# This is a natural language question:\n"
code_sol_intro = "# Here is the python code to solve this question:\n"
code_sol_outro = "# End of the code\n"

def text_to_code_prompting(text_list: List[str], code_list: List[str], test_text: str) -> str:
    assert len(text_list) == len(code_list)

    prompt_example_strs = []
    for text, code in zip(text_list, code_list):
        prompt_example_strs.append(nl_intro + text + "\n\n" + code_sol_intro + code + "\n" + code_sol_outro + "\n")

    final_prompt = task_description + "\n\n" + "".join(prompt_example_strs) + \
                   nl_intro + test_text + "\n\n" + code_sol_intro

    return final_prompt


def mathqa_get_pre_answer_state(example) -> str:
    # skip the last state, which is the one containing the answer already
    for state_dict in example["states"][::-1][1:]:
        if state_dict["type"] == "stmt":
            return f"# {'; '.join([f'{key} = {val}' for key, val in state_dict['execution_state'].items()])}\n"
        else:
            continue


def mathqa_answer_prompting(fewshot_examples: List[Dict[str, Any]], test_example: Dict[str, Any], 
                            answer_state: bool = False, add_gold_answer: bool = False) -> Tuple[str, str]:
    test_text, test_code = test_example["text"], test_example["code"]

    prompt_example_strs = []
    for example in fewshot_examples:
        text, code = example["text"], example["code"]

        # get the answer state and incorporate in the code
        answer_state_str = mathqa_get_pre_answer_state(example) if answer_state else ""
        answer_state_str += f"# answer = {example['answer']}\n" if add_gold_answer else ""
        answer_stmt_idx = code.find("answer = ")
        code = code[:answer_stmt_idx] + answer_state_str + code[answer_stmt_idx:]

        prompt_example_strs.append(nl_intro + text + "\n\n" + code_sol_intro + code + "\n" + code_sol_outro + "\n")

    prompt = task_description + "\n\n" + "".join(prompt_example_strs) + \
                   nl_intro + test_text + "\n\n" + code_sol_intro

    # add the prompt for the test example
    context_sol_code = test_code[:test_code.find("answer = ")]
    context_sol_code += mathqa_get_pre_answer_state(test_example) if answer_state else ""
    context_sol_code += f"# answer = {test_example['answer']}\n" if add_gold_answer else ""
    context_sol_code += "answer = "
    prompt += context_sol_code

    return prompt, context_sol_code




    
    



    