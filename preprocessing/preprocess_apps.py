import sys
sys.path.append('/home/t-ansongni/Code/trace-codegen/')

import json
import os
import random
from tqdm import tqdm

from typing import Dict, Any, List
from lightning_modules.datasets.reader_utils import get_statements_from_code, byte_idx_to_char_idx
from tree_sitter import Language, Parser

DATA_DIR = 'data/apps'
SAVE_DATA_DIR = 'data/apps/preprocessed'

SHARD_NUM = 16
VAL_SET_SIZE = 512

if not os.path.exists(SAVE_DATA_DIR):
    os.makedirs(SAVE_DATA_DIR)


# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__), 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def load_instance(dir_path: str) -> Dict[str, Any]:
    instance_dict = {}

    # for each folder, there are four files: `input_output.json`, `metadata.json`, `question.txt`, `solutions.json`, `starter_code.py`
    metadata_file = os.path.join(dir_path, 'metadata.json')
    with open(metadata_file, 'r') as f:
        instance_dict['metadata'] = json.load(f)

    question_file = os.path.join(dir_path, 'question.txt')
    with open(question_file, 'r') as f:
        instance_dict['question'] = f.read()

    # the following two are optional for training
    input_output_file = os.path.join(dir_path, 'input_output.json')
    if os.path.exists(input_output_file):
        with open(input_output_file, 'r') as f:
            instance_dict['input_output'] = json.load(f)

    starter_code_file = os.path.join(dir_path, 'starter_code.py')
    if os.path.exists(starter_code_file):
        with open(starter_code_file, 'r') as f:
            instance_dict['starter_code'] = f.read()
        instance_dict['problem_type'] = 'call_based'
    else:
        instance_dict['problem_type'] = 'standard_input'

    # solutions are optional for test instances
    solutions_file = os.path.join(dir_path, 'solutions.json')
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            instance_dict['solutions'] = []
            solutions = json.load(f)
            for solution in solutions:
                stmts = get_statements_from_code(solution, parser)
                if stmts is None:
                    continue
                instance_dict['solutions'].append({'raw_code': solution, 'stmts': stmts})
                    

    return instance_dict


if __name__ == '__main__':
    dataset_dict: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_name in ['train', 'test']:
        print(f"load and process dataset: {dataset_name}")
        instances = []
        for i in tqdm(range(5000)):
            num_str = f"{i:04d}"
            instance_dir = os.path.join(DATA_DIR, dataset_name, num_str)
            instance_json = load_instance(instance_dir)
            instances.append(instance_json)

        if dataset_name == 'train':
            # hold out for the validation set
            # validation set must all have input/outputs
            io_instances = list(filter(lambda x: 'input_output' in x, instances))
            no_io_instances = list(filter(lambda x: 'input_output' not in x, instances))
            random.shuffle(io_instances)

            dataset_dict['val'] = io_instances[:VAL_SET_SIZE]
            dataset_dict['train'] = no_io_instances + io_instances[VAL_SET_SIZE:]
            random.shuffle(dataset_dict['train'])
        else:
            dataset_dict['test'] = instances

    for dataset_name, instances in dataset_dict.items():
        print(f"sharding and saving the instances: {dataset_name}")
        # save the instances
        save_file_name = os.path.join(SAVE_DATA_DIR, f'{dataset_name}.jsonl')
        with open(save_file_name, 'w+') as f:
            for ins in instances:
                f.write(json.dumps(ins) + '\n')
            print(f"saved {len(instances)} instances to {save_file_name}")

        # further save in shards
        shard_dir = os.path.join(SAVE_DATA_DIR, f'{dataset_name}_shards')
        if not os.path.exists(shard_dir):
            os.makedirs(shard_dir)

        if dataset_name == 'train':
            # a more clever binning to make the shard size more balanced
            shards = [[] for _ in range(SHARD_NUM)]
            stmts_num = [(0, i) for i in range(SHARD_NUM)]

            for ins in instances:
                size = sum([len(x['stmts']) for x in ins['solutions']])
                min_idx = min(stmts_num, key=lambda x: x[0])[1]
                shards[min_idx].append(ins)
                stmts_num[min_idx] = (stmts_num[min_idx][0] + size, min_idx)
            print(f"the total stmts num in the shards are {stmts_num}")
        else:
            shards = [instances[i::SHARD_NUM] for i in range(SHARD_NUM)]

        for i in range(SHARD_NUM):
            shard_file_name = os.path.join(shard_dir, f'{dataset_name}_shard_{i}.jsonl')
            with open(shard_file_name, 'w+') as f:
                for ins in shards[i]:
                    f.write(json.dumps(ins) + '\n')
                print(f"saved {len(shards[i])} instances to {shard_file_name}")