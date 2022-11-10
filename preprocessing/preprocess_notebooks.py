import os
import nbformat
import random
import json

from nbformat.validator import validate
from itertools import chain
from tqdm import tqdm
from typing import List, Tuple, Union, Any, Dict
from nbformat.reader import NotJSONError
from multiprocessing import Process, Manager
from tree_sitter import Language, Parser
from allennlp_modules.dataset_readers.reader_utils import get_statements_from_code


VAL_SET_PERCENT = 0.02 # set the validation set size to a relative percentage number 
VAL_SET_SIZE = 100*1000 # set the validation set size to a fixed number (instead of percentage)


def _load_all_notebooks_from_dir(nbs_dir: str) -> List[str]:
    dataset_dirs = [os.path.join(nbs_dir, d) for d in os.listdir(nbs_dir) if os.path.isdir(os.path.join(nbs_dir, d))]

    # list all the notebooks in the dataset
    nb_paths = []
    for dataset_dir in dataset_dirs:
        dataset_nbs = [f for f in os.listdir(dataset_dir) if f.endswith('.ipynb')]
        dedupped_nbs = dict()
        for nb_name in dataset_nbs:
            stemmed_nb_name = '_'.join(nb_name.split('_')[2:])
            # TODO: deduping: now just taking the first notebook, but we can use other criteria, like length, etc
            if stemmed_nb_name not in dedupped_nbs:
                dedupped_nbs[stemmed_nb_name] = nb_name
        nb_paths.extend(list(os.path.join(dataset_dir, f) for f in dedupped_nbs.values()))

    return nb_paths


def _get_cells(nb_path):
    """identify the code cells and markdown cells and extract the content"""
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except NotJSONError:
        print('Notebook {} is not a JSON file'.format(nb_path))
        print('Skipping notebook...')
        return []

    # extract code from the cells
    try:
        cells = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                cells.append({'idx': i, 'type': 'code', 'lines': cell.source.strip().split('\n')})
            elif cell.cell_type == 'markdown':
                cells.append({'idx': i, 'type': 'markdown', 'lines': cell.source.strip().split('\n')})
            elif cell.cell_type == 'raw':
                # TODO : raw cells, what do they do?
                pass
            else:
                print(f'unexpected cell type: {cell.cell_type}')
        return cells
    except Exception as e:
        print(f'Error reading notebook {nb_path}, skip...')
        print(e)
        return []


def _process_notebooks_in_dir(nbs_dir: str, result_list: List):
    nb_paths: List[str] = _load_all_notebooks_from_dir(nbs_dir)
    for nb_path in tqdm(nb_paths):
        nb_cells = _get_cells(nb_path)
        if len(nb_cells) > 0:
            result_list.append({"path": nb_path, "cells": nb_cells})


def parse_cell(lines: List[str], py_parser: Parser) -> Tuple[str, List[Dict]]:
    """ parse the cell lines into scopes and statements with tree-sitter """
    # remove notebook lines such as `%load_ext`
    processed_lines = list(filter(lambda x: not (x.startswith('%') or x.startswith('!')), lines))
    cell_code = '\n'.join(processed_lines)

    target_stmts = get_statements_from_code(cell_code, py_parser, tolerate_errors=False)

    return processed_lines, target_stmts


def load_notebook_data(data_dir: str) -> List[Dict[str, Any]]:
    """ load all the notebook cells from all the original notebooks """
    nbs_dir_base = os.path.join(data_dir, 'original_nb_')
    nbs_dirs = [nbs_dir_base + str(i) for i in range(2, 10)]

    with Manager() as manager:
        result_list = manager.list()
        processes = []
        for i in range(len(nbs_dirs)):
            p = Process(target=_process_notebooks_in_dir, args=(nbs_dirs[i], result_list))
            print(f"loading from dir {nbs_dirs[i]} with process {p}")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return list(result_list)


if __name__ == '__main__':
    data_dir = "/home/t-ansongni/trace_nbblob/notebooks"
    save_data_dir = "data/original_nb_stmt"

    all_notebooks = load_notebook_data(data_dir)

    # parse all the cells to find the scopes and statements
    language_build_path = os.path.join(os.path.dirname(__file__), 'py-tree-sitter.so')
    PY_LANGUAGE = Language(language_build_path, 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    parsed_notebooks = []
    for i, ins in tqdm(enumerate(all_notebooks)):
        cells = ins['cells']
        nb_path = ins['path']

        parseable = True
        for cell in cells:
            if cell['type'] == 'code':
                processed_lines, stmt_list = parse_cell(cell['lines'], parser)
                cell['lines'] = processed_lines
                if stmt_list is not None:
                    cell['stmts'] = stmt_list
                else:
                    # drop this notebook if there is a cell that can't be parsed
                    parseable = False
                    break
        if parseable:
            parsed_notebooks.append(ins)
        # else:
        #     print(f"Drop unparseable notebook {nb_path}")
    print(f"{len(all_notebooks) - len(parsed_notebooks)} notebooks are dropped since they contain unparseable cells")
    all_notebooks = parsed_notebooks

    # group the notebooks by datasets
    dataset_nb_dict = dict()
    for notebook_result in all_notebooks:
        nb_path = notebook_result['path']
        data_set_name = nb_path.split('/')[-2]

        if data_set_name not in dataset_nb_dict:
            dataset_nb_dict[data_set_name] = [notebook_result]
        else:
            dataset_nb_dict[data_set_name].append(notebook_result)
    all_datasets = list(dataset_nb_dict.items())

    # split into train and validation set based on datasets (instead of notebooks or cells) to avoid too much overlap
    print(f"{len(all_datasets)} datasets in total")
    random.shuffle(all_datasets)
    train_datasets = all_datasets[:int(len(all_datasets) * (1.0-VAL_SET_PERCENT))]
    val_datasets = all_datasets[int(len(all_datasets) * (1.0-VAL_SET_PERCENT)):]

    train_instances = list(chain(*[x[1] for x in train_datasets]))
    val_instances = list(chain(*[x[1] for x in val_datasets]))

    train_cell_num = sum(len(nb['cells']) for nb in train_instances)
    val_cell_num = sum(len(nb['cells']) for nb in val_instances)

    train_stmt_num = sum(sum(len(cell['stmts']) if 'stmts' in cell else 0 
                            for cell in nb['cells']) for nb in train_instances)
    val_stmt_num = sum(sum(len(cell['stmts']) if 'stmts' in cell else 0 
                            for cell in nb['cells']) for nb in val_instances)

    print(f"{len(train_datasets)} datasets for training, {len(val_datasets)} datasets for validation")
    print(f"{len(train_instances)} notebooks for training, {len(val_instances)} notebooks for validation")
    print(f"{train_cell_num} cells for training, {val_cell_num} cells for validation")
    print(f"{train_stmt_num} target stmts for training, {val_stmt_num} target stmts for validation")

    # save the data
    with open(os.path.join(save_data_dir, 'train.json'), 'w+') as f:
        json.dump(train_instances, f)
    with open(os.path.join(save_data_dir, 'val.json'), 'w+') as f:
        json.dump(val_instances, f)
    