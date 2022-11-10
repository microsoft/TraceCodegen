import multiprocessing
import multiprocessing.pool
import ast
import math

from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor as Pool
from execution.safe_execution_util import execute

######################################################################################################
# following are some dataset specific functions for getting the execution result
######################################################################################################

def mathqa_answer_eq(prediction: Any, gold_answer: Any):
    try:
        # if the execution result is a numpy array, valueError will be raised
        if math.isclose(float(prediction), float(gold_answer), abs_tol=1e-4):
            return True
        else:
            return False
    except (ValueError, TypeError, OverflowError):
        return False

def mathqa_execution(program: str) -> Any:
    """
    for mathqa-python, we should be getting the answers from the "answer" variable in the local() variables
    """

    result = execute(program)
    
    if result["result"] == "passed":
        if "answer" in result["locals"]:
            executed_answer = result["locals"]["answer"]
        else:
            # FIXME: this is so ad-hoc
            executed_answer = -10000
    else:
        executed_answer = None

    return executed_answer


######################################################################################################
# following are different metrics for evaluating the execution result
# FIXME: Right now we only consider single test cases
######################################################################################################

def batch_exec_programs(programs: List[str], exec_func: callable, n_processes: int = 20) -> List[Any]:
    # build a dict to optimize for potential same programs
    program_dict = {}
    for program in programs:
        if program not in program_dict:
            program_dict[program] = None
    unique_programs = list(program_dict.keys())

    idx = 0
    parsable_unique_programs = []
    for program in unique_programs:
        try:
            ast.parse(program, mode="exec")
            parsable_unique_programs.append(program)
            program_dict[program] = idx
            idx += 1
        except SyntaxError:
            program_dict[program] = -1
        except MemoryError:
            print(f"MemoryError when parsing {program}")
            program_dict[program] = -1
        except ValueError:
            print(f"ValueError when parsing {program}")
            program_dict[program] = -1

    with Pool(n_processes) as p:
        unique_executed_answers = p.map(exec_func, parsable_unique_programs)
    unique_executed_answers = list(unique_executed_answers)
    unique_executed_answers.append(None) # all syntax error will be assigned to None

    # build the original programs answer list
    executed_answers = [unique_executed_answers[program_dict[program]] for program in programs]

    return executed_answers, len(unique_programs)

def batch_execution_acc(programs: List[str], exec_func: callable, answers: List[str], 
                        n_examples: int, eval_at_k: int, n_processes: int = 20) -> List[Tuple[float, float]]:
    """
    This function evaluates execution accuracy for a batch of programs using multiprocessing.

    Returns: execution accuracy, execution rate
    """
    assert len(programs) == len(answers) * eval_at_k
    assert n_examples * eval_at_k == len(programs)

    executed_answers, n_unique_programs = batch_exec_programs(programs, exec_func, n_processes)
    print(f"Evaluating {len(programs)} generated programs for {n_examples} tasks, " + \
           f"but only {n_unique_programs} unique programs")
    pct_unique_programs = n_unique_programs / len(programs)

    # separate the results for each task
    grouped_executed_answers = [executed_answers[i*eval_at_k:(i+1)*eval_at_k] for i in range(0, n_examples)]
    grouped_execution_evals = []
    for predicted_answers, gold_answer in zip(grouped_executed_answers, answers):
        correct_count = 0.0
        for predicted_answer in predicted_answers:
            if mathqa_answer_eq(predicted_answer, gold_answer):
                correct_count += 1

        accuracy_at_k = correct_count / eval_at_k
        pass_at_k = correct_count > 0.0

        grouped_execution_evals.append((accuracy_at_k, pass_at_k))

    return grouped_execution_evals, pct_unique_programs

def execution_acc(program: str, exec_func: callable, answer: str) -> Tuple[float, float]:
    """
    This function is used to evaluate the accuracy of the execution of the program.

    Returns: execution accuracy, execution rate
    """
    executed_answer = exec_func(program)
    if executed_answer is not None and mathqa_answer_eq(executed_answer, answer):
        return 1.0, 1.0
    elif executed_answer is not None:
        return 0.0, 1.0
    else:
        return 0.0, 0.0

def execution_eval_at_k(programs: List[str], exec_func: callable, answer: str, k: int) -> Tuple[float, float]:
    """
    Assign 1.0 when at least one out of the k programs execute to the correct answer

    Returns: (accuracy_at_k, pass_at_k)
    """
    assert len(programs) >= k, "The number of programs should be larger than k"

    correct_count = 0.0
    with Pool(20) as p:
        executed_answers = p.map(exec_func, programs[:k])
    for executed_answer in executed_answers:
        if mathqa_answer_eq(executed_answer, answer):
            correct_count += 1

    accuracy_at_k = correct_count / k
    pass_at_k = correct_count > 0.0

    return accuracy_at_k, pass_at_k
