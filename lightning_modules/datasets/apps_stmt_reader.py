import json
import pathlib
import re
import logging
import sys
import numpy as np
import re
import os
import torch

from overrides import overrides
from typing import Dict, Iterable, List, Any, Optional, Union
from functools import reduce
from tqdm import tqdm
from itertools import chain

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from .reader_utils import last_byte_idx_to_token_idx, END_OF_CELL_TOKEN
from lightning_modules.models.gpt_util import get_gpt

from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader

# set environment variable to avoid deadlocks, see: 
# https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader.common_issues
os.environ['TOKENIZERS_PARALLELISM']='0'

logger = logging.getLogger(__name__)
SHARD_NUM = 16

class AppsStmtDataset(Dataset):

    def __init__(
        self, 
        file_path: List[str],
        transformer_model_name: str, 
        min_context_stmts: int, 
        max_context_tokens: int,
        max_target_tokens: int,
        max_stmt_tokens: int,
        min_stmt_tokens: int,
        max_sol_tokens: int,
        problem_as_unit: bool,
        allow_skip: bool,
        max_instances: int,
        **kwargs):
        super().__init__(**kwargs)
        _, self.tokenizer = get_gpt(transformer_model_name, additional_special_tokens = [END_OF_CELL_TOKEN])

        self._min_context_stmts = min_context_stmts
        self._max_context_tokens = max_context_tokens
        self._max_target_tokens = max_target_tokens
        self._max_stmt_tokens = max_stmt_tokens
        self._min_stmt_tokens = min_stmt_tokens
        self._max_sol_tokens = max_sol_tokens
        self.problem_as_unit = problem_as_unit
        self.allow_skip = allow_skip
        self.max_instances = max_instances

        self.instances = []
        for p in file_path:
            self.instances.extend(self.read(p))

    def read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        print("Reading dataset files at %s", file_path)

        count_dict = {'total_yield_instances': 0,
                      'skipped_instances': 0,
                      'cutoff_context_instances': 0}

        all_yield_instances = []

        with open(file_path, 'r') as f:
            def _load_problem():
                while True:
                    line = f.readline()
                    if not line:
                        break

                    problem = json.loads(line)
                    problem_instances = self.load_instances_from_problem(problem, count_dict)

                    for ins in problem_instances:
                        yield ins
            
            for ins in tqdm(_load_problem()):
                all_yield_instances.append(ins)
                if len(all_yield_instances) >= self.max_instances:
                    break

            print(f"{count_dict} from {file_path}")

        return all_yield_instances

    def load_instances_from_problem(self, problem: Dict[str, Any], 
                                    count_dict: Dict[str, int]) -> Iterable[Dict[str, Any]]:
        all_problem_instances = []

        question = problem['question']
        question_tokens = self.tokenizer.tokenize(question) + [self.tokenizer.eos_token]
        starter_code = problem.get('starter_code', None)
        starter_code_tokens = self.tokenizer.tokenize(starter_code) + [self.tokenizer.eos_token] if starter_code else None

        problem_type = problem['problem_type']
        input_outputs = problem.get('input_output', None)
        solutions = problem['solutions']

        # starter_code can only be missing in standard input type of problems
        assert (starter_code is not None) or (problem_type == 'standard_input')
        # input_outputs can only be missing in train set
        assert (input_outputs is not None) or not self.problem_as_unit

        # process the tokens for each of the solutions
        all_sol_stmt_tokens: List[List[List[str]]] = []

        # print(f"num solutions: {len(solutions)}")
        for i, sol in enumerate(solutions):
            sol_raw_str = sol['raw_code']
            sol_tokens = self.tokenizer.tokenize(sol_raw_str)
            
            if len(sol_tokens) > self._max_sol_tokens:
                print(f"Skipping solution of length {len(sol_tokens)} tokens")
                continue

            # assert starter_code_tokens is None or \
            #     starter_code_tokens == sol_tokens[:len(starter_code_tokens)]

            # first split the cell by stmts
            sol_stmts = sol['stmts']
            # DEBUG setting: make sure that the stmts are actually sorted
            assert all(sol_stmts[j]['end_byte'] < sol_stmts[j+1]['end_byte'] for j in range(len(sol_stmts)-1))

            # then split the stmts by 'end_byte' => end_token
            end_token_indices = [0]+[last_byte_idx_to_token_idx(stmt['end_byte']-1, sol_tokens, self.tokenizer) + 1
                                        for stmt in sol_stmts]
            stmt_tokens = [sol_tokens[stmt_idx_start:stmt_idx_end] 
                            for (stmt_idx_start, stmt_idx_end) 
                                in zip(end_token_indices[:-1], end_token_indices[1:])]

            # add the eos to every stmt
            for tokens in stmt_tokens:
                tokens.append(self.tokenizer.eos_token)
            
            # add a special end of cell token as the last stmt
            last_stmt_tokens = sol_tokens[end_token_indices[-1]:] \
                                + [END_OF_CELL_TOKEN, self.tokenizer.eos_token]
            stmt_tokens.append(last_stmt_tokens)
            sol_stmts.append({'type': 'eoc'})
            assert len(sol_stmts) == len(stmt_tokens)

            all_sol_stmt_tokens.append(stmt_tokens)
        
        if len(all_sol_stmt_tokens) == 0:
            print("skipping problems without solutions since we do not have execution accuracy yet...")
            # assert self.problem_as_unit, "this should only happen for a val/test set"
            return []

        # at inference time, the problem will be emitted as a whole and all solutions will be included as multiple references
        if self.problem_as_unit:
            context_tokens = (question_tokens + (starter_code_tokens if starter_code else []))
            if len(context_tokens) > self._max_context_tokens:
                count_dict['cutoff_context_instances'] += 1
                context_tokens = context_tokens[-self._max_context_tokens:]

            # NOTE: the starter code may in both the context and the target, but this is okay since we only use 
            # the target tokens for evaluation of ROUGE, etc.
            target_tokens = [list(reduce(lambda x, y: x + y, sol_stmt_tokens)) for sol_stmt_tokens in all_sol_stmt_tokens]

            metadata = {'problem_type': problem_type, 'question': question, 
                        'input_outputs': input_outputs, 'starter_code': starter_code,
                        'pad_token': self.tokenizer.pad_token_id}

            instance = self.tokens_to_instance(context_tokens, target_tokens, metadata) 
            all_problem_instances.append(instance)
            count_dict['total_yield_instances'] += 1
        else:
            for sol_stmts in all_sol_stmt_tokens:
                # at training time, we fit as much stmts as possible in the target window
                stmt_idx = 0
                # stmt_tokens = sol_stmts[stmt_idx]
                while stmt_idx < len(sol_stmts):
                    if stmt_idx < self._min_context_stmts and self.allow_skip:
                        count_dict['skipped_instances'] += 1
                        stmt_idx += 1
                        continue

                    # NOTE: at training time, the starter code does not matter because solution already includes the starter code
                    context_tokens = question_tokens + (list(reduce(lambda x,y: x+y, sol_stmts[:stmt_idx])) 
                                                                                if stmt_idx > 0 else [])
                    if len(context_tokens) > self._max_context_tokens:
                        count_dict['cutoff_context_instances'] += 1
                        context_tokens = context_tokens[-self._max_context_tokens:]

                    # assemble the target upto max target tokens
                    target_tokens = list(sol_stmts[stmt_idx])
                    while stmt_idx + 1 < len(sol_stmts) and \
                        len(target_tokens) + len(sol_stmts[stmt_idx+1]) <= self._max_target_tokens:
                        stmt_idx += 1
                        target_tokens.extend(sol_stmts[stmt_idx])

                    # this is necessary because the first stmt might already be too long
                    target_tokens = target_tokens[:self._max_target_tokens] 

                    metadata = {'problem_type': problem_type, 'question': question, 
                                'input_outputs': input_outputs, 'starter_code': starter_code,
                                'pad_token': self.tokenizer.pad_token_id}

                    instance = self.tokens_to_instance(context_tokens, [target_tokens], metadata)
                    all_problem_instances.append(instance)
                    count_dict['total_yield_instances'] += 1
                    stmt_idx += 1 # remove this will create a dead loop under some conditions
        
        return all_problem_instances

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)


    def tokens_to_instance(self, context_tokens: List[str], target_tokens: List[List[str]], 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ use this to assume the lines are already tokenized, to avoid repeated tokenization when using a sliding window """

        # currently the implementation is towards GPT-Neo
        assert self.tokenizer.name_or_path.startswith("EleutherAI")

        # multiple targets is only for the inference time
        assert self.problem_as_unit or len(target_tokens) == 1

        # build the metadata field, which is common for training and inference
        instance = {'metadata': metadata}
        instance['metadata']['context_lines'] = self.tokenizer.convert_tokens_to_string(context_tokens).split('\n')

        # some common fields for training and inference
        input_tokens = context_tokens + target_tokens[0] 
        instance['input_tokens'] = torch.tensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        instance['target_mask'] = torch.tensor([0]*len(context_tokens) + [1]*len(target_tokens[0]))
        instance['input_mask'] = torch.tensor([1]*len(input_tokens))

        if self.problem_as_unit: # inference
            instance['context_tokens'] = torch.tensor(self.tokenizer.convert_tokens_to_ids(context_tokens))
            instance['context_mask'] = torch.tensor([1]*len(context_tokens))
            instance['metadata']['target_str'] = [self.tokenizer.convert_tokens_to_string(target) 
                                                    for target in target_tokens]
        else: # training
            instance['metadata']['target_str'] = [self.tokenizer.convert_tokens_to_string(target_tokens[0])]

        return instance

def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}

    pad_token_id = examples[0]['metadata']['pad_token']

    for k in examples[0].keys():
        if k == 'metadata':
            result_dict[k] = [ex[k] for ex in examples]
        elif k.endswith('_mask'):
            result_dict[k] = torch.nn.utils.rnn.pad_sequence([ex[k] for ex in examples], batch_first=True).bool()
        elif k.endswith('_tokens'):
            result_dict[k] = torch.nn.utils.rnn.pad_sequence([ex[k] for ex in examples], batch_first=True,
                                                             padding_value=pad_token_id).long()

    return result_dict

class AppsDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                batch_size: int, 
                val_batch_size: int,
                train_file_path: str,
                val_file_path: str,
                test_file_path: str = None,
                min_context_stmts: int = 3,
                max_context_tokens: int = 412,
                max_target_tokens: int = 256,
                max_stmt_tokens: int = 100,
                min_stmt_tokens: int = 0,
                max_sol_tokens: int = 10000,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = transformer_model_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

        self.min_context_stmts = min_context_stmts
        self.max_context_tokens = max_context_tokens
        self.max_target_tokens = max_target_tokens
        self.max_stmt_tokens = max_stmt_tokens
        self.min_stmt_tokens = min_stmt_tokens
        self.max_sol_tokens = max_sol_tokens

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # TODO: temporarily using this to read from individual shards
        assert stage in ["fit", "validate", "test"]

        if torch.distributed.is_initialized() and False:
            # determine num of files to read per GPU
            process_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            assert SHARD_NUM % world_size == 0, f"# gpus ({world_size}) must be a multiple of shard num ({SHARD_NUM})"
            file_per_gpu = SHARD_NUM // world_size

            train_files = [os.path.join(self.train_file_path, f"train_shard_{process_rank+i}.jsonl") 
                                for i in range(file_per_gpu)]
            val_files = [os.path.join(self.val_file_path, f"val_shard_{process_rank+i}.jsonl") 
                                for i in range(file_per_gpu)]
            if self.test_file_path:
                test_files = [os.path.join(self.test_file_path, f"test_shard_{process_rank+i}.jsonl") 
                                    for i in range(file_per_gpu)]
        else:
            train_files = [self.train_file_path]
            val_files = [self.val_file_path]
            if self.test_file_path:
                test_files = [self.test_file_path]

        nb_cell_stmt_train = AppsStmtDataset(file_path=train_files,
                                                transformer_model_name=self.transformer_model_name,
                                                min_context_stmts=self.min_context_stmts,
                                                max_context_tokens=self.max_context_tokens,
                                                max_target_tokens=self.max_target_tokens,
                                                max_stmt_tokens=self.max_stmt_tokens,
                                                min_stmt_tokens=self.min_stmt_tokens,
                                                max_sol_tokens=self.max_sol_tokens,
                                                max_instances=self.train_max_instances,
                                                problem_as_unit=False,
                                                allow_skip=True)

        # deal with the uneven number of instances across shards in distributed training
        if torch.distributed.is_initialized():
            self.even_shards(nb_cell_stmt_train)
        self.train_data = nb_cell_stmt_train

        nb_cell_stmt_val = AppsStmtDataset(file_path=val_files,
                                                transformer_model_name=self.transformer_model_name,
                                                min_context_stmts=self.min_context_stmts,
                                                max_context_tokens=self.max_context_tokens,
                                                max_target_tokens=self.max_target_tokens,
                                                max_stmt_tokens=self.max_stmt_tokens,
                                                min_stmt_tokens=self.min_stmt_tokens,
                                                max_sol_tokens=self.max_sol_tokens,
                                                max_instances=self.val_max_instances,
                                                problem_as_unit=True,
                                                allow_skip=False)
        self.val_data = nb_cell_stmt_val

    def even_shards(self, shard_data: AppsStmtDataset):
        assert torch.distributed.is_initialized(), "this is only needed for distributed training"

        # calculate the minimum of all shards
        world_size = torch.distributed.get_world_size()
        instance_num_list = [-1 for _ in range(world_size)]
        torch.distributed.all_gather_object(instance_num_list, len(shard_data))
        min_instance_num = min(instance_num_list)
        assert min_instance_num > 0

        # truncate all shards to the minimum
        truncated_instances = shard_data.truncate(int(min_instance_num))
        all_truncated_instances = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_truncated_instances, truncated_instances)
        all_truncated_instances = list(chain.from_iterable(all_truncated_instances))

        # redistributed the truncated portion
        extended_instances = []
        process_rank = torch.distributed.get_rank()
        for i in range(len(all_truncated_instances)):
            if i % world_size == process_rank:
                extended_instances.append(all_truncated_instances[i])
        shard_data.extend(extended_instances)

    # return the dataloader for each split
    def train_dataloader(self):
        mnist_train = DataLoader(self.train_data, batch_size=self.batch_size, 
                                 collate_fn=customized_collate_fn, shuffle=True, drop_last=True)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.val_data, batch_size=self.val_batch_size, 
                               collate_fn=customized_collate_fn, shuffle=False, drop_last=True)
        return mnist_val

    def test_dataloader(self):
        raise NotImplementedError