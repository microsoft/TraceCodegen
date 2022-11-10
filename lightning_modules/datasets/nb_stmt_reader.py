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
from typing import Dict, Iterable, List, Tuple, Any, Optional
from functools import reduce
from tqdm import tqdm

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

class NbCellStmtDataset(Dataset):

    def __init__(
        self, 
        file_path: str,
        transformer_model_name: str, 
        max_context_stmts: int, 
        min_context_stmts: int, 
        max_context_tokens: int,
        max_target_tokens: int,
        max_stmt_tokens: int,
        min_stmt_tokens: int,
        cell_as_unit: bool,
        allow_skip: bool,
        max_instances: int,
        **kwargs):
        super().__init__(**kwargs)
        _, self.tokenizer = get_gpt(transformer_model_name, additional_special_tokens = [END_OF_CELL_TOKEN])

        self._max_context_stmts = max_context_stmts
        self._min_context_stmts = min_context_stmts
        self._max_context_tokens = max_context_tokens
        self._max_target_tokens = max_target_tokens
        self._max_stmt_tokens = max_stmt_tokens
        self._min_stmt_tokens = min_stmt_tokens
        self.cell_as_unit = cell_as_unit
        self.allow_skip = allow_skip
        self.max_instances = max_instances

        self.instances = self.read(file_path)

    def read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        print("Reading dataset files at %s", file_path)

        count_dict = {'total_yield_instances': 0,
                      'skipped_instances': 0}

        all_yield_instances = []

        with open(file_path, 'r') as f:
            def _load_notebook():
                while True:
                    line = f.readline()
                    if not line:
                        break

                    notebook = json.loads(line)
                    notebook_instances = self.load_instances_from_notebook(notebook, count_dict)
                    for ins in notebook_instances:
                        yield ins
            
            for ins in tqdm(_load_notebook()):
                all_yield_instances.append(ins)
                if len(all_yield_instances) >= self.max_instances:
                    break

            print(f"{count_dict} from {file_path}")

        return all_yield_instances

    def load_instances_from_notebook(self, notebook: Dict[str, Any], 
                                     count_dict: Dict[str, int]) -> Iterable[Dict[str, Any]]:
        all_notebook_instances = []

        nb_path = notebook['path']
        nb_cells = notebook['cells']

        all_cell_raw_strs: List[str] = []
        all_cell_tokens: List[List[str]] = []

        for cell in nb_cells:
            # preprocess the cells based on the type
            if cell['type'] != 'code':
                cell_raw_str = "\n".join([('# '+line) for line in cell['lines']])
            else:
                cell_raw_str = "\n".join(cell['lines'])

            all_cell_raw_strs.append(cell_raw_str)
            cell_tokens = self.tokenizer.tokenize(cell_raw_str)
            all_cell_tokens.append(cell_tokens)

        all_cell_stmt_tokens: List[List[List[str]]] = []

        for i, cell in enumerate(nb_cells):
            if cell['type'] == 'code':
                # first split the cell by stmts
                cell_stmts = cell['stmts']
                # DEBUG setting: make sure that the stmts are actually sorted FIXME: this assert actually breaks sometime, find out why
                # assert all(cell_stmts[i]['end_byte'] < cell_stmts[i+1]['end_byte'] for i in range(len(cell_stmts)-1))
                cell_stmts = sorted(cell_stmts, key=lambda x: x['end_byte'])

                # then split the stmts by 'end_byte' => end_token
                end_token_indices = [0]+[last_byte_idx_to_token_idx(stmt['end_byte']-1, all_cell_tokens[i], self.tokenizer) + 1
                                            for stmt in cell_stmts]
            else:
                # simply split by the \n characters
                all_before_newline_bytes = [i-1 for i, x in enumerate(bytes(all_cell_raw_strs[i], 'utf-8')) if x == ord('\n')]
                end_token_indices = [0]+[last_byte_idx_to_token_idx(byte_idx, all_cell_tokens[i], self.tokenizer) + 1
                                            for byte_idx in all_before_newline_bytes]
                cell_stmts = []

            stmt_tokens = [all_cell_tokens[i][stmt_idx_start:stmt_idx_end] 
                            for (stmt_idx_start, stmt_idx_end) 
                                in zip(end_token_indices[:-1], end_token_indices[1:])]
            for tokens in stmt_tokens:
                tokens.append(self.tokenizer.eos_token)
            
            # add a special end of cell token as the last stmt
            last_stmt_tokens = all_cell_tokens[i][end_token_indices[-1]:] \
                                + [END_OF_CELL_TOKEN, self.tokenizer.eos_token]
            stmt_tokens.append(last_stmt_tokens)
            cell_stmts.append({'type': 'eoc'})
            assert cell['type'] == 'markdown' or len(cell_stmts) == len(stmt_tokens)

            all_cell_stmt_tokens.append(stmt_tokens)

            if not cell['type'] == 'code':
                # if the cell is not code, we will not emit any instances
                continue
            
            if not self.cell_as_unit:
                # at training time, we fit as much stmts as we can in the target window
                stmt_idx = 0
                while stmt_idx < len(stmt_tokens):
                    if self.allow_skip and \
                        stmt_idx + sum(len(st) for st in all_cell_stmt_tokens[:i]) < self._min_context_stmts:
                        count_dict['skipped_instances'] += 1
                        stmt_idx += 1
                        continue

                    # assemble the context upto max context stmts
                    context_stmts = stmt_tokens[:stmt_idx]
                    for j in range(i-1, -1, -1):
                        context_stmts = all_cell_stmt_tokens[j] + context_stmts
                        if len(context_stmts) >= self._max_context_stmts:
                            context_stmts = context_stmts[-self._max_context_stmts:]
                            break
                    context_tokens = [token for st in context_stmts for token in st][-self._max_context_tokens:]

                    # assemble the target upto max target tokens
                    target_tokens = list(stmt_tokens[stmt_idx])
                    while stmt_idx + 1 < len(stmt_tokens) and \
                        len(target_tokens) + len(stmt_tokens[stmt_idx+1]) <= self._max_target_tokens:
                        stmt_idx += 1
                        target_tokens.extend(stmt_tokens[stmt_idx])

                    # this is necessary because the first stmt might already be too long
                    target_tokens = target_tokens[:self._max_target_tokens] 

                    instance = self.tokens_to_instance(context_tokens, target_tokens,
                                                        nb_path+f"[cell{cell['idx']}]", 
                                                        statement_type='multiple_stmts')
                    # yield instance
                    all_notebook_instances.append(instance)
                    count_dict['total_yield_instances'] += 1
                    stmt_idx += 1
            else:
                # at inference time, we emit the all the stmts in the cell as a whole
                if self.allow_skip and sum(len(st) for st in all_cell_stmt_tokens[:i]) < self._min_context_stmts:
                    count_dict['skipped_instances'] += 1
                    continue

                # the target tokens are all the tokens in the cell
                target_tokens = [token for st in stmt_tokens for token in st]

                # assemble the context upto max context stmts
                context_stmts = []
                for j in range(i-1, -1, -1):
                    context_stmts = all_cell_stmt_tokens[j] + context_stmts
                    if len(context_stmts) >= self._max_context_stmts:
                        context_stmts = context_stmts[-self._max_context_stmts:]
                        break
                context_tokens = [token for st in context_stmts for token in st][-self._max_context_tokens:]

                instance = self.tokens_to_instance(context_tokens, target_tokens,
                                                    nb_path+f"[cell{cell['idx']}]", 
                                                    statement_type='whole_cell')
                # yield instance
                all_notebook_instances.append(instance)
                count_dict['total_yield_instances'] += 1

        return all_notebook_instances

    def __getitem__(self, idx: int):
        # since not every process will read the same amount of instances (we can't guarantee that)
        idx = idx % len(self.instances)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)


    def tokens_to_instance(self, context_tokens: List[str], target_tokens: List[str], 
                           nb_path: str, statement_type: str) -> Dict[str, Any]:
        """ use this to assume the lines are already tokenized, to avoid repeated tokenization when using a sliding window """

        # currently the implementation is towards GPT-Neo
        assert self.tokenizer.name_or_path.startswith("EleutherAI")
        # input tokens are used in training, and it's the only place to add the EOS token at the end for learning
        input_tokens = context_tokens + target_tokens
        output_mask = torch.tensor([0]*len(context_tokens) +
                               [1]*len(target_tokens))
        input_mask = torch.tensor([1]*len(input_tokens))
        context_mask = torch.tensor([1]*len(context_tokens))

        assert len(output_mask) == len(input_tokens)

        # add the strings for context and target
        context_str = self.tokenizer.convert_tokens_to_string([str(token) for token in context_tokens])
        target_str = self.tokenizer.convert_tokens_to_string([str(token) for token in target_tokens])

        instance = {"input_tokens": torch.tensor(self.tokenizer.convert_tokens_to_ids(input_tokens)),
                    "target_mask": output_mask,
                    "input_mask": input_mask,
                    "context_mask": context_mask,
                    "context_tokens": torch.tensor(self.tokenizer.convert_tokens_to_ids(context_tokens)),
                    "target_tokens": torch.tensor(self.tokenizer.convert_tokens_to_ids(target_tokens)),
                    "metadata": {"nb_path": nb_path, 
                                 "target_str": target_str, 
                                 "context_lines": context_str.split('\n'), 
                                 "stmt_type": statement_type,
                                 "pad_token": self.tokenizer.pad_token_id}}
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

class NbCellDataModule(LightningDataModule):
    def __init__(self, 
                batch_size: int, 
                val_batch_size: int,
                train_file_path: str,
                val_file_path: str,
                test_file_path: str = None,
                max_context_stmts: int = 100,
                min_context_stmts: int = 3,
                max_context_tokens: int = 412,
                max_target_tokens: int = 412,
                max_stmt_tokens: int = 100,
                min_stmt_tokens: int = 0,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

        self.max_context_stmts = max_context_stmts
        self.min_context_stmts = min_context_stmts
        self.max_context_tokens = max_context_tokens
        self.max_target_tokens = max_target_tokens
        self.max_stmt_tokens = max_stmt_tokens
        self.min_stmt_tokens = min_stmt_tokens

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # TODO: temporarily using this to read from individual shards
        assert stage in ["fit", "validate"]

        if torch.distributed.is_initialized():
            process_rank = torch.distributed.get_rank()
            self.train_file_path = os.path.join(self.train_file_path, f"train_shard_{process_rank}.jsonl")
            self.val_file_path = os.path.join(self.val_file_path, f"val_shard_{process_rank}.jsonl")
            if self.test_file_path:
                self.test_file_path = os.path.join(self.test_file_path, f"test_shard_{process_rank}.jsonl")

        nb_cell_stmt_train = NbCellStmtDataset(file_path=self.train_file_path,
                                                transformer_model_name="EleutherAI/gpt-neo-125M",
                                                max_context_stmts=self.max_context_stmts,
                                                min_context_stmts=self.min_context_stmts,
                                                max_context_tokens=self.max_context_tokens,
                                                max_target_tokens=self.max_target_tokens,
                                                max_stmt_tokens=self.max_stmt_tokens,
                                                min_stmt_tokens=self.min_stmt_tokens,
                                                max_instances=self.train_max_instances,
                                                cell_as_unit=False,
                                                allow_skip=True)
        self.train_data = nb_cell_stmt_train
        nb_cell_stmt_val = NbCellStmtDataset(file_path=self.val_file_path,
                                                transformer_model_name="EleutherAI/gpt-neo-125M",
                                                max_context_stmts=self.max_context_stmts,
                                                min_context_stmts=self.min_context_stmts,
                                                max_context_tokens=self.max_context_tokens,
                                                max_target_tokens=self.max_target_tokens,
                                                max_stmt_tokens=self.max_stmt_tokens,
                                                min_stmt_tokens=self.min_stmt_tokens,
                                                max_instances=self.val_max_instances,
                                                cell_as_unit=True,
                                                allow_skip=False)
        self.val_data = nb_cell_stmt_val

    # return the dataloader for each split
    def train_dataloader(self):
        nb_train = DataLoader(self.train_data, batch_size=self.batch_size, 
                                 collate_fn=customized_collate_fn, shuffle=True, drop_last=True)
        return nb_train

    def val_dataloader(self):
        nb_val = DataLoader(self.val_data, batch_size=self.val_batch_size, 
                               collate_fn=customized_collate_fn, shuffle=False, drop_last=True)
        return nb_val

    def test_dataloader(self):
        raise NotImplementedError