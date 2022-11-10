import torch
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

from matplotlib.axes import Axes
from tqdm import tqdm
from typing import List, Dict, Any
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from lightning_modules.models.gpt_seq2seq_model import GptSeq2SeqModel

DEVICE = "cuda:0"
MODEL_PATH = "/home/v-ansongni/Code/trace-codegen/amlt/mathqa-dedup-partial-mml-len_norm-low-temp/gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/checkpoints/step=55755-exec_acc=0.1263-exec_rate=0.9727.ckpt"
# MODEL_PATH = "/home/v-ansongni/Code/trace-codegen/amlt/mathqa-dedup-mml-low-temp/gpt-neo-mathqa-state-finetuning/lightning_logs/version_0/checkpoints/step=68475-exec_acc=0.1302-exec_rate=0.9780.ckpt"
# MODEL_PATH = "/home/v-ansongni/Code/trace-codegen/amlt/mathqa-dedup-mle-low-temp-r/gpt-neo-mathqa-finetuning/lightning_logs/version_0/checkpoints/step=38159-exec_acc=0.1159-exec_rate=0.9844.ckpt"
DATA_FILE = "./data/mathqa/train_dedup.jsonl"
# BUFFER_FILES = [f"amlt/mathqa-dedup-mml-low-temp/gpt-neo-mathqa-state-finetuning/" \
#                 f"lightning_logs/version_0/buffer_step_68687_rank_{i}.jsonl" for i in range(16)]
BUFFER_FILES = [f"amlt/mathqa-dedup-partial-mml-len_norm-low-temp/gpt-neo-mathqa-state-finetuning/" \
                f"lightning_logs/version_0/buffer_step_61055_rank_{i}.jsonl" for i in range(16)]

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def get_buffered_examples(examples: List[Dict[str, Any]], buffer_examples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    # build the buffer dict
    buffer_dict = {}
    for example in buffer_examples:
        buffer_dict[example["task_id"]] = example

    # for every example, emit a list of examples each with the saved fcp program in the buffer
    result = []
    for example in examples:
        task_id = example["task_id"]
        example_list = []
        for i, fcp in enumerate(buffer_dict[task_id]["saved_fcp_programs"]):
            example_dict = example.copy()
            example_dict["code"] = fcp
            example_dict["task_id"] = example_dict["task_id"] + f"_{i}"
            example_list.append(example_dict)
        result.append(example_list)
            
    return result

def visualize_logits(logits: torch.Tensor, correct_token_idx: torch.Tensor, example_id: str, context_len: int,
                     correct_prog_log_prob: float, tokenizer: GPT2Tokenizer, top_k: int = 5):
    assert logits.shape[0] == 1
    log_probs = log_softmax(logits[0][context_len:].detach(), dim=-1)

    assert correct_token_idx.shape[0] == 1
    correct_token_idx = correct_token_idx[0][context_len:]

    # get the top k from the logits
    top_k_results = log_probs.topk(top_k, dim = -1)
    top_k_values = top_k_results[0].transpose(0, 1)
    top_k_indices = top_k_results[1].transpose(0, 1)

    # from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html 
    ranks = ["1", "2", "3", "4", "5"]
    correct_tokens = tokenizer.convert_ids_to_tokens(correct_token_idx.tolist())[1:]

    top_tokens = [tokenizer.convert_ids_to_tokens(x) for x in top_k_indices.tolist()]
    top_token_probs = torch.exp(top_k_values).cpu().numpy()


    fig, ax = plt.subplots(figsize=(len(top_tokens[0]), len(top_tokens)))
    im = ax.imshow(top_token_probs, aspect='auto', cmap=plt.get_cmap('coolwarm'), interpolation='none')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(correct_tokens)), labels=correct_tokens, size=8)
    ax.set_yticks(np.arange(len(ranks)), labels=ranks, size=8)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ranks)):
        for j in range(len(correct_tokens)):
            token = top_tokens[i][j]
            token_prob = top_token_probs[i][j]
            if token == correct_tokens[j]:
                text = ax.text(j, i, f"{token}: {token_prob:.2f}",
                            ha="center", va="center", color="black", size=8, weight="bold")
            else:
                text = ax.text(j, i, f"{token}: {token_prob:.2f}",
                            ha="center", va="center", color="w", size=8)

    ax.set_title(f"Top-5 token probabilities, whole program prob: {np.exp(correct_prog_log_prob):.4f}")
    # fig.tight_layout()
    plt.savefig(f"prog_graph_vis/pred_{example_id}_vis.png", dpi=100)
    plt.cla()
    plt.clf()

def tokenizer_math_qa(tokenizer: GPT2Tokenizer, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    tokenizer_outputs = tokenizer("\n".join([example["text"], example["code"]]))
    context_len = len(tokenizer.encode(example["text"]+"\n"))

    example_dict = {}
    example_dict["input_ids"] = torch.tensor([tokenizer_outputs["input_ids"] + [tokenizer.eos_token_id]], device=DEVICE)
    example_dict["attention_mask"] = torch.tensor([tokenizer_outputs["attention_mask"] + [1]], device=DEVICE)
    example_dict["labels"] = torch.tensor([[-100] * context_len + tokenizer_outputs["input_ids"][context_len:] + [tokenizer.eos_token_id]], device=DEVICE)
    example_dict["task_id"] = example["task_id"]
    example_dict["context_len"] = context_len
    

    return example_dict


def run_model_inference(model: GptSeq2SeqModel, examples: List[Dict[str, Any]]):
    assert "input_ids" in examples[0] and "attention_mask" in examples[0] and "labels" in examples[0]

    log_probs = []
    for example in examples:
        input_ids, attention_mask, labels = example["input_ids"], example["attention_mask"], example["labels"]
        gpt_result = model.gpt(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

        # get the prediction logits, shape: (batch_size, seq_length, vocab_size)
        shift_logits = gpt_result.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flatten_unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        unreduced_loss =flatten_unreduced_loss.view(shift_labels.shape) * attention_mask[..., 1:]
        loss = unreduced_loss.sum() # we need log prob of the sequence

        log_prob = -float(loss)
        log_probs.append(log_prob)

        # vis
        # visualize_logits(shift_logits, correct_token_idx=input_ids, example_id=example["task_id"], 
        #                  context_len = example["context_len"]-1, # to account for the first token
        #                  correct_prog_log_prob=-float(loss), 
        #                  tokenizer=model.tokenizer)
    
    return log_probs

def main():
    # init the model
    model = GptSeq2SeqModel(transformer_model_name = "EleutherAI/gpt-neo-125M", max_gen_len = 256, 
                            load_ckpt_file=MODEL_PATH, 
                            optimizer={"init_args": {"lr": 1e-4, "betas": [0.9, 0.999], 
                                                     "eps": 1e-8, "weight_decay": 0.1}}).to(DEVICE)

    # read the data
    train_examples = read_jsonl_file(DATA_FILE)
    buffer_examples = read_jsonl_file(BUFFER_FILES[0])

    vis_examples = get_buffered_examples(train_examples, buffer_examples)

    FIRST_K_GROUP = 5
    prob_groups = [[] for _ in range(FIRST_K_GROUP)]
    for example_list in tqdm(vis_examples[:200]):
        # print(f"{len(example_list)} examples in the fcp buffer")
        example_dict_list = [tokenizer_math_qa(model.tokenizer, example) for example in example_list]
        log_probs = run_model_inference(model, example_dict_list)

        if len(log_probs) == 1:
            continue
        else:
            probs = [math.exp(x) for x in log_probs]
            probs = sorted(probs, reverse=True)
            print(probs)
            for i, prob in enumerate(probs[:FIRST_K_GROUP]):
                prob_groups[i].append(prob)

    for i, prob_group in enumerate(prob_groups):
        print(f"group-{i}: mean {np.mean(prob_group):.4f}; std {np.std(prob_group):.4f}")

    # plot using scatter
    xs = []
    ys = []
    for i, prob_group in enumerate(prob_groups):
        for j, prob in enumerate(prob_group):
            xs.append(i)
            ys.append(prob)
    plt.scatter(xs, ys)
    plt.savefig(f"prob_scatter_partial_mml.png", dpi=100)


if __name__ == "__main__":
    main()