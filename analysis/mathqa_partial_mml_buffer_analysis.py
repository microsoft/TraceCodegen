import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List, Dict, Any, Tuple
from itertools import chain

from analysis.analysis_utils import binning
from execution.program_tracing import get_execution_states, batch_program_tracing
from execution.program_tracing import Program, ProgTrace, ProgTraceUnit, ProgState, HashableProgState

from lightning_modules.models.gpt_stmt_partial_mml_model import get_hashable_state, mathqa_identify_fully_correct_state
from lightning_modules.models.gpt_seq2seq_model import post_process_code

training_file = "./data/mathqa/train_dedup.jsonl"
# prediction_files = [f"amlt/mathqa-partial-mml-first_try-r/gpt-neo-mathqa-state-finetuning/" \
#                 f"lightning_logs/version_0/predictions_step_13210_rank_{i}.jsonl" for i in range(16)]
# buffer_files = [f"amlt/mathqa-mml-fixed_loss-max_marg_10-r/gpt-neo-mathqa-state-finetuning/" \
#                 f"lightning_logs/version_0/buffer_step_45599_rank_{i}.jsonl" for i in range(16)]
buffer_files = [f"amlt/mathqa-dedup-partial-mml-fixed_dedup-no_sampling_from_state-r/gpt-neo-mathqa-state-finetuning/" \
                f"lightning_logs/version_0/buffer_step_40374_rank_{i}.jsonl" for i in range(16)]

def vis_states_as_graph(gold_state: str, fcp_states: List[str], pcp_states: List[str], 
                        gold_program_lines: List[str], fcp_lines: List[List[str]], pcp_lines: List[List[str]], 
                        example_id: str = 'tmp'):
    G = nx.Graph()

    # add nodes
    all_states_strs = [gold_state] + fcp_states + pcp_states
    all_nodes_set = set(chain.from_iterable(all_states_strs))
    G.add_nodes_from(list(all_nodes_set))

    # add edges
    edge_labels = {}
    def add_edge_from_state(state_str: str, c: str, lines: List[str]):
        lines = list(filter(lambda l: l.strip() != "", lines))
        while not len(state_str) == len(lines) + 1:
            # sometimes we have the first statement that doesn't produce the state
            state_str = '*' + state_str
        existing_edges = list(G.edges)
        for i in range(len(state_str)-1):
            if (state_str[i], state_str[i+1]) not in existing_edges and (state_str[i+1], state_str[i]) not in existing_edges:
                G.add_edge(state_str[i], state_str[i+1], color=c, weight=2, minlen=len(lines[i]) * 100)
                edge_labels[(state_str[i], state_str[i+1])] = f"{c}: {lines[i]}"
            else:
                edge_labels[(state_str[i], state_str[i+1])] += "\n" + f"{c}: {lines[i]}"

    add_edge_from_state(gold_state, "g", gold_program_lines)
    for i, s in enumerate(fcp_states):
        add_edge_from_state(s, "r", fcp_lines[i])
    for i, s in enumerate(pcp_states):
        add_edge_from_state(s, "b", pcp_lines[i])

    # visualize
    pos = nx.spring_layout(G)
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    # weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G, pos, edge_color=colors, with_labels=True, font_weight='bold', node_color='yellow')
    # nx.draw(G, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=4, 
                                 verticalalignment='baseline')
    plt.savefig(f"prog_graph_vis/{example_id}_graph_vis.png", dpi=300)
    plt.clf()

def saved_programs_length_analysis(buffer: List[Dict[str, Any]], training_data: Dict[str, Any]):
    all_fcp_length_diffs = []
    all_pcp_length_diffs = []

    for instance in buffer:
        pcp_programs = instance["saved_pcp_programs"]
        fcp_programs = instance["saved_fcp_programs"]

        train_json_instance = training_data[instance["task_id"]]
        gold_program = train_json_instance["code"]

        assert gold_program in fcp_programs, f"{gold_program} not in fcp_programs"
        assert "" in pcp_programs, "empty program not in pcp_programs"

        # calculate the length excepting the intitial programs (gold and empty)
        fcp_length_diffs = [len(p.split("\n")) - len(gold_program.split("\n")) for p in filter(lambda p: p != gold_program, fcp_programs)]
        pcp_length_diffs = [len(p.split("\n")) - len(gold_program.split("\n")) for p in filter(lambda p: p != "", pcp_programs)]

        all_fcp_length_diffs.extend(fcp_length_diffs)  
        all_pcp_length_diffs.extend(pcp_length_diffs)

    # use the histogram plot
    binning([-8, -4, -2, -1, 0, 1, 2, 4, 8], all_fcp_length_diffs, title="FCP length diffs in buffer (not normalized)")
    binning([-8, -4, -2, -1, 0, 1, 2, 4, 8], all_pcp_length_diffs, title="PCP length diffs in buffer (not normalized)")

def saved_program_state_detour_analysis(buffer: List[Dict[str, Any]], training_data: Dict[str, Any]):
    for instance in tqdm(buffer[:100]):
        train_json_instance = training_data[instance["task_id"]]
        gold_program = train_json_instance["code"]
        # gold_program = post_process_code(train_json_instance["code"])

        pcp_programs = list(filter(lambda x: x != "", instance["saved_pcp_programs"]))
        fcp_programs = list(filter(lambda x: x != gold_program, instance["saved_fcp_programs"]))
        # pcp_programs = []
        # fcp_programs = list(filter(lambda x: x != gold_program, instance["saved_programs"]))

        # check for duplicates after post-processing
        all_programs = list(filter(lambda x: len(x) > 0, [gold_program] + fcp_programs + pcp_programs))
        if len(all_programs) != len(set(all_programs)) or len(all_programs) != len(set([post_process_code(p) for p in all_programs])):
            for p in all_programs:
                print(p + "\n" + "=================")

        all_program_traces = batch_program_tracing(all_programs)

        gold_program_trace: ProgTrace = all_program_traces[0]
        gold_state_list = list(filter(lambda x: len(x) > 0, [get_hashable_state(unit.state) 
                                                if len(unit.state) > 0 else "" for unit in gold_program_trace]))

        def index_state(trace_unit: ProgTraceUnit) -> str:
            if len(trace_unit.state) == 0:
                return ""

            if mathqa_identify_fully_correct_state(trace_unit.state, train_json_instance["answer"]):
                return '$'
            
            hashable_state = get_hashable_state(trace_unit.state)

            try:
                idx = gold_state_list.index(hashable_state)
            except ValueError:
                idx = len(gold_state_list)
                gold_state_list.append(hashable_state)

            return chr(ord('A') + idx)

        trace_reprs = ["".join([index_state(unit) for unit in trace]) for trace in all_program_traces]
        trace_reprs = ['*' + repr for repr in trace_reprs]

        vis_states_as_graph(trace_reprs[0], trace_reprs[1:len(fcp_programs)+1], 
                            trace_reprs[len(fcp_programs)+1:], 
                            [unit.code for unit in all_program_traces[0]], 
                            [[unit.code for unit in program_trace] for program_trace in all_program_traces[1:len(fcp_programs)+1]],
                            [[unit.code for unit in program_trace] for program_trace in all_program_traces[len(fcp_programs)+1:]],
                            example_id=instance["task_id"])

        print("")


def analyze_buffer(buffer: List[Dict[str, Any]]):
    for instance in buffer:

        print("")

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    training_data = read_jsonl_file(training_file)
    training_data_dict = {instance["task_id"]: instance for instance in training_data}

    # since the buffers are synced by the end of the training epoch, we only need to load one buffer
    buffer = read_jsonl_file(buffer_files[0])

    # sort the buffer
    buffer = sorted(buffer, key=lambda x: x["task_id"])

    # saved_programs_length_analysis(buffer, training_data_dict)

    saved_program_state_detour_analysis(buffer, training_data_dict)
