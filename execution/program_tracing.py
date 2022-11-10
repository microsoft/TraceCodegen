import math
import scipy
import json
import os

from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List, Dict, Tuple, Any, Union, NamedTuple, Set
from scipy import special

from typing import List, Dict, Any
from tqdm import tqdm
from lightning_modules.datasets.reader_utils import get_statements_from_code, byte_idx_to_char_idx
from execution.safe_execution_util import execute, canonicalize_var_dict
from tree_sitter import Language, Parser

ProgState = Dict[str, float]
HashableProgState = Tuple[str]
ProgTraceUnit = NamedTuple("ProgTraceUnit", [("code", str), ("type", str), ("state", ProgState)])
ProgTrace = List[ProgTraceUnit]
Program = NamedTuple("Program", [("code", str), ("code_lite", str), ("trace", ProgTrace)])

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__)+'/../preprocessing/', 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

"""
Tracing the execution of a program:
    1. It parses the program into a sequence of tracing units (currently stmts);
    2. Make some markings of the tracing units;
    3. Insert tracing code to the program, after every tracing unit;
    4. Run the program with tracing;
    5. Collect the variable tracing information.
"""

from copy import deepcopy
from types import ModuleType

tracing_local_list = []
def record_state(local_var_dict):
    copied_local_var_dict = canonicalize_var_dict(local_var_dict)
    tracing_local_list.append(copied_local_var_dict)

def get_execution_states(program: str, debug=False) -> Union[ProgTrace, None]:
    # first parse the program with tree-sitter
    stmts = get_statements_from_code(program, parser)

    if stmts is None:
        if debug:
            print(f'skipping unparseable example')
            print(f"##########\n{program}\n##########")
        return None

    # extract the stmt strings
    idx = 0
    stmt_states = []
    for stmt in stmts:
        start_idx = byte_idx_to_char_idx(stmt['start_byte'], program)
        end_idx = byte_idx_to_char_idx(stmt['end_byte'], program)

        if start_idx != idx:
            # add the gap
            stmt_states.append({"code": program[idx:start_idx], "type": "gap"})

        # add the stmt
        stmt_states.append({"code": program[start_idx:end_idx], "type": "stmt"})
        idx = end_idx


    # NOTE: FIXME: This only works for straight-line programs since it does not consider indentation
    for stmt in stmt_states:
        if stmt["type"] == "stmt":
            stmt["code"] += "\nrecord_state(locals())"

    # now assemble the program back together
    traced_program = "".join([stmt["code"] for stmt in stmt_states])

    # execute the program with tracing code
    result = execute(traced_program, {}, 
                     globals={"tracing_local_list": tracing_local_list, "deepcopy": deepcopy, 
                              "record_state": record_state, "ModuleType": ModuleType,
                              "math": math, "scipy": scipy, "scipy.special": special}, use_tracing=True)

    if result["result"] == "passed":
        # add the *output* states for each statement and remove the tracing code to restore orginal program
        stmt_idx = 0
        for stmt in stmt_states:
            if stmt["type"] == "stmt":
                stmt["execution_state"] = result["tracing_local_list"][stmt_idx]
                stmt["code"] = stmt["code"].replace("\nrecord_state(locals())", "")
                stmt_idx += 1
        prog_trace = [ProgTraceUnit(stmt["code"], stmt["type"], 
                        stmt["execution_state"] if stmt["type"] == "stmt" else {}) for stmt in stmt_states]
        return prog_trace
    else:
        if debug:
            print(f'skipping example of error: {result["result"]}')
            print(f"##########\n{program}\n##########")
        return None

def batch_program_tracing(programs: List[str], n_processes=20) -> List[Union[ProgTrace, None]]:
    with Pool(n_processes) as p:
        tracing_outputs = p.map(get_execution_states, programs)
    return list(tracing_outputs)

def exec_stmt_in_context(stmt: str, context: Dict[str, Any]):
    # NOTE: FIXME: This only works for straight-line programs since it does not consider indentation
    traced_stmt = stmt + "\nrecord_state(locals())"

    # execute the program with tracing code
    if "math" in context:
        context["math"] = math
    if "scipy" in context:
        context["scipy"] = scipy
        context["scipy.special"] = special

    result = execute(traced_stmt, locals=context, 
                     globals={"tracing_local_list": tracing_local_list, "deepcopy": deepcopy, 
                              "record_state": record_state, "ModuleType": ModuleType}, use_tracing=True)

    if result["result"] == "passed":
        # return the execution states as the local var list
        assert len(result["tracing_local_list"]) == 1, f"tracing_local_list: {result['tracing_local_list']}"
        stmt_output_state = result["tracing_local_list"][0]
        return stmt_output_state
    else:
        return None

def is_trivial_state(state_dict: Dict[str, Any], prev_stmt: str):
    if len(state_dict) == 0:
        return True

    assert prev_stmt is not None, "prev_stmt must be provided to determine trivial states unless the state is empty"

    if prev_stmt.split(" ")[0] in ["#", "import"]:
        return True

    assert len(state_dict) == 1, f"prev_stmt {prev_stmt}; original state dict {state_dict}"

    return f"{list(state_dict.keys())[0]} = {list(state_dict.values())[0]}" in prev_stmt

def get_state_repr(state_dict: Dict[str, Any], prev_stmt: str = None, only_include_keys: List[str] = None, 
                   prev_state_dict: Dict[str, Any] = None, use_diff=False, skip_trivial_states: bool = False):
    if use_diff:
        raise NotImplementedError

    if only_include_keys is not None:
        state_dict = {k: v for k, v in state_dict.items() if k in only_include_keys}

    if skip_trivial_states and is_trivial_state(state_dict, prev_stmt):
        return ""

    repr = "# "
    for key, value in state_dict.items():
        repr += f"{key} = {value}; "
    repr += "\n"

    return repr

if __name__ == "__main__":
    # load some sample programs
    with open('data/mathqa/val-python.jsonl', 'r') as f:
        lines = f.readlines()

        json_examples = [json.loads(line) for line in lines]

    with open('data/mathqa/val_python_with_states.jsonl', 'w+') as f:
        success_count = 0
        failed_count = 0
        for json_example in tqdm(json_examples):

            program = json_example["code"]
            stmt_states = get_execution_states(program)

            if stmt_states is not None:
                json_example["states"] = stmt_states
                f.write(json.dumps(json_example) + "\n")
                success_count += 1
            else:
                failed_count += 1

        print(f"Successfully traced {success_count}/{success_count+failed_count} programs")