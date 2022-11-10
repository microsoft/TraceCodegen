import json
import editdistance
import numpy as np

from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from bashplotlib.histogram import plot_hist

from analysis.analysis_utils import train_file_path, test_file_path, val_file_path, load_examples, binning

def text_word_edit_dist(dict_1: Dict[str, Any], dict_2: Dict[str, Any]) -> int:
    """
    Calculate the edit distance between two texts.
    """
    text_1 = dict_1['text']
    text_2 = dict_2['text']
    return editdistance.eval(text_1.split(" "), text_2.split(" "))  

def measure_and_save_sim_matrix(
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    save_path: str
) -> None:
    """
    Calculate the similarity matrix between train and test data.
    """
    sim_matrix = np.zeros((len(train_data), len(test_data)))
    for i, train_dict in enumerate(tqdm(train_data)):
        for j, test_dict in enumerate(test_data):
            sim_matrix[i, j] = text_word_edit_dist(train_dict, test_dict)
    np.save(save_path, sim_matrix)

def load_and_analysis_sim_matrix(
    src_dict_list: List[Dict[str, Any]],
    tgt_dict_list: List[Dict[str, Any]],
    save_path: str
) -> None:
    """
    Load similarity matrix from file and analysis the result.
    """
    sim_matrix = np.load(save_path)

    # exclude itself in computing the most similar example when using the same set
    if len(src_dict_list) == len(tgt_dict_list):
        np.fill_diagonal(sim_matrix, np.inf)

    min_list = np.min(sim_matrix, axis=0) 
    argmin_list = np.argmin(sim_matrix, axis=0)

    for i, dict_2 in enumerate(tgt_dict_list[:10]):
        if min_list[i] > 10:
            continue 
        dict_1 = src_dict_list[argmin_list[i]]
        print("################################")
        print(f"{dict_1['text']}")
        print("--------------------------------")
        print(f"{dict_2['text']}")
        print("--------------------------------")
        print(f"edit distance is {min_list[i]}")
        print("################################")

    binning([0, 2, 4, 8, 16], min_list)

def get_overlap_example_ids(set_name: str, min_allow_dist: int) -> List[int]:
    assert set_name in ['train', 'test', 'val']

    save_path = f"analysis/train_{set_name}_overlap.npy"
    sim_matrix = np.load(save_path)

    # exclude itself in computing the most similar example when using the same set
    if set_name == 'train':
        np.fill_diagonal(sim_matrix, np.inf)

    min_list = np.min(sim_matrix, axis=0) 

    overlapping_ids = []
    for i, min_dist in enumerate(min_list):
        if min_dist > min_allow_dist:
            continue
        else:
            overlapping_ids.append(i)

    return overlapping_ids

def filter_out_overlapping_examples(set_name: str, min_allow_dist: int) -> List[Dict[str, Any]]:
    assert set_name in ['train', 'test', 'val']

    if set_name == 'train':
        src_dict_list = load_examples(train_file_path)
    elif set_name == 'test':
        src_dict_list = load_examples(test_file_path)
    elif set_name == 'val':
        src_dict_list = load_examples(val_file_path)

    overlapping_ids = get_overlap_example_ids(set_name, min_allow_dist)

    filtered_dict_list = list(map(lambda y: y[1], filter(lambda x: x[0] not in overlapping_ids, enumerate(src_dict_list))))

    print(f"{len(src_dict_list)} -> {len(filtered_dict_list)} examples after filtering out overlapping examples")

    with open(f"data/mathqa/filtered_{min_allow_dist}_{set_name}_with_state.jsonl", "w+") as f:
        for dict_ in filtered_dict_list:
            f.write(json.dumps(dict_) + "\n")
        print(f"Saved filtered {set_name} data to {f.name}")

if __name__ == "__main__":
    # measure the overlap between train and test data
    train_data = load_examples("data/mathqa/filtered_2_train_with_state.jsonl")
    val_data = load_examples("data/mathqa/filtered_2_val_with_state.jsonl")
    test_data = load_examples("data/mathqa/filtered_2_test_with_state.jsonl")
    # measure_and_save_sim_matrix(train_data, val_data, "analysis/filtered_train_val_overlap.npy")

    load_and_analysis_sim_matrix(train_data, val_data, "analysis/filtered_train_val_overlap.npy")

    # filter_out_overlapping_examples("train", 2)
    # filter_out_overlapping_examples("test", 2)
    # filter_out_overlapping_examples("val", 2)
