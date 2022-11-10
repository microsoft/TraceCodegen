import torch
import time
import os
import json
import random

from itertools import chain
from concurrent.futures import ProcessPoolExecutor as Pool

from typing import Optional, Dict, Any, Tuple, List, Set, Union
from torch.nn import CrossEntropyLoss

from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule

from .gpt_stmt_state_model import GptStmtStateModel 
from .gpt_seq2seq_model import post_process_code
from .gpt_util import left_pad_sequences
from execution.execution_evaluation import execution_acc, mathqa_execution, batch_exec_programs, mathqa_answer_eq


class GptStmtMmlModel(GptStmtStateModel):
    def __init__(self, 
                 transformer_model_name: str,
                 load_gold_programs: bool = True,
                 on_policy_sample_num: Union[int, float] = 5,
                 on_policy_sample_temp: float = 0.8,
                 max_sampling_len: int = 100,
                 length_diff_tolerance: int = 100,
                 marg_set_size: int = 100,
                 max_buffer_size: int = 100,
                 load_samples_file: str = None,
                 exclude_context_loss: bool = False,
                 beta_smoothing: float = 1.0,
                 mle_lambda: float = 1.0,
                 mml_lambda: float = 0.0,
                 mle_aug_norm: bool = False,
                 **kwargs) -> None:
        if "eval_with_states" in kwargs and kwargs["eval_with_states"]:
            raise ValueError("eval_with_states is not supported for GptStmtMmlModel")

        super().__init__(transformer_model_name, **kwargs)

        self.load_gold_programs = load_gold_programs
        self.on_policy_sample_num = on_policy_sample_num
        self.on_policy_sample_temp = on_policy_sample_temp
        self.max_sampling_len = max_sampling_len
        self.marg_set_size = marg_set_size
        self.max_buffer_size = max_buffer_size
        self.length_diff_tolerance = length_diff_tolerance
        self.exclude_context_loss = exclude_context_loss
        self.beta_smoothing = beta_smoothing
        assert self.beta_smoothing > 0.0 and self.beta_smoothing <= 1.0, \
            f"beta_smoothing must be in (0, 1], but got {self.beta_smoothing}"
        self.mle_lambda = mle_lambda
        self.mml_lambda = mml_lambda
        self.mle_aug_norm = mle_aug_norm

        # load the large sample of programs from the buffer
        self.loaded_samples: Dict[str, List[str]] = {}
        if load_samples_file is not None:
            self.load_samples_from_file(load_samples_file)
            print(f"loaded samples from {load_samples_file}")

        # the key being the task id and the value being the list of correct programs (in strs, *w/o* eos token)
        self.correct_program_buffer: Dict[str, Set[str]] = {} 

        # define some debugging or eval metrics
        self.metrics_dict["pct_unique_programs"] = MeanMetric()

    def load_samples_from_file(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            for line in f:
                json_dict = json.loads(line)
                self.loaded_samples[json_dict["metadata"]["task_id"]] = json_dict["generated_k_programs"]

    def try_save_programs(self, generated_programs: List[str], task_ids: List[str], 
                          correct_answers: List[float], samples_per_task: int, 
                          log_unique_program_pct: bool = True, verbose: bool = False) -> None:
        # execute the programs first
        program_exec_results, n_unique_programs = batch_exec_programs(generated_programs, mathqa_execution)#, n_processes=5)
        if log_unique_program_pct:
            self.metrics_dict["pct_unique_programs"](n_unique_programs / (len(task_ids) * samples_per_task))

        # save the programs with correct results into the buffer if it's not already in there
        correct_count = 0
        saved_count = 0
        for i, exec_result in enumerate(program_exec_results):
            example_idx = i // samples_per_task
            if mathqa_answer_eq(exec_result, correct_answers[example_idx]):
                correct_count += 1
                task_id = task_ids[example_idx]
                generated_program = post_process_code(generated_programs[i], ast_back_parse=False)
                if generated_program not in self.correct_program_buffer[task_id]:
                    # check whether satisfied the length difference tolerence
                    min_prog_len = min([len(prog.split("\n")) for prog in self.correct_program_buffer[task_id]])
                    if len(generated_program.split("\n")) - min_prog_len <= self.length_diff_tolerance:
                        # save the program into the buffer
                        self.correct_program_buffer[task_id].add(generated_program)
                        saved_count += 1
        if verbose:
            print(f"{len(generated_programs)} in total, {n_unique_programs} are unique, " + \
                    f"{correct_count} programs are correct, saved {saved_count} programs into the buffer")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # use task ids as the identifier
        task_ids = [ex["task_id"] for ex in batch["metadata"]]

        # add the gold programs to the buffer
        for i, example in enumerate(batch["metadata"]):
            task_id = task_ids[i]
            if task_id in self.correct_program_buffer:
                # already have loaded the correct programs for this task
                continue
            elif self.load_gold_programs:
                gold_program = example["code"]
                # the program in the buffer is always post-processed
                self.correct_program_buffer[task_id] = set((post_process_code(gold_program, ast_back_parse=False),))

                # load the program from the samples, if available
                if task_id in self.loaded_samples:
                    loaded_examples_for_task = self.loaded_samples.pop(task_id) # this also removes the key
                    self.try_save_programs(loaded_examples_for_task, [task_id], [batch["metadata"][i]["answer"]], 
                                           len(loaded_examples_for_task), log_unique_program_pct=False)
                    
            else:
                self.correct_program_buffer[task_id] = set()
        
        # do on-policy sampling for the current tasks
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.on_policy_sample_num > 0:
            with torch.no_grad():
                if not all([len(self.correct_program_buffer[task_id]) >= self.max_buffer_size for task_id in task_ids]):
                    # generate the programs and get their execution results
                    max_context_len = input_ids.size(1)
                    output_seqs = self.gpt.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                                    do_sample=True, max_new_tokens=self.max_sampling_len,
                                                    num_return_sequences=self.on_policy_sample_num,
                                                    temperature=self.on_policy_sample_temp)
                    generated_seqs = output_seqs[:, max_context_len:].cpu()
                    generated_programs = self.tokenizer.batch_decode(generated_seqs, skip_special_tokens=True)

                    # try to save the programs
                    correct_answers = [x["answer"] for x in batch["metadata"]]
                    self.try_save_programs(generated_programs, task_ids, correct_answers, self.on_policy_sample_num)

        def sample_from_buffer(task_id: str) -> List[str]:
            cached_programs = list(self.correct_program_buffer[task_id])
            if len(cached_programs) <= self.marg_set_size:
                return cached_programs
            else:
                return random.sample(cached_programs, self.marg_set_size)

        # remove the left paddings first and concat the context and cached programs
        context_seqs = [input_ids[i, -context_len:] for i, context_len in enumerate(attention_mask.sum(dim=1))]
        cached_program_seqs: List[List[torch.Tensor]] = [[self.tokenizer(prog_str, return_tensors="pt")['input_ids'][0]
                                                            for prog_str in sample_from_buffer(task_id)] 
                                                                for task_id in task_ids]
        cached_program_nums = [len(cached_programs) for cached_programs in cached_program_seqs]
        flatten_input_ids = []
        for i, program_seqs in enumerate(cached_program_seqs):
            for program_seq in program_seqs:
                flatten_input_ids.append(torch.cat((context_seqs[i], program_seq.to(self.device),
                                            torch.tensor([self.tokenizer.eos_token_id], device=self.device)), dim=0))
        flatten_attention_mask = [torch.ones_like(flatten_ids) for flatten_ids in flatten_input_ids]
        flatten_input_ids = left_pad_sequences(flatten_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        flatten_attention_mask = left_pad_sequences(flatten_attention_mask, batch_first=True, padding_value=0)

        # exclude the loss from context by setting the labels of them to be -100
        if self.exclude_context_loss:
            flatten_labels = []
            for i, program_seqs in enumerate(cached_program_seqs):
                for j, program_seq in enumerate(program_seqs):
                    concat_labels = torch.cat((-100 * torch.ones_like(context_seqs[i]), program_seq.to(self.device),
                                                torch.tensor([self.tokenizer.eos_token_id], device=self.device)), dim=0)
                    flatten_labels.append(concat_labels)
            flatten_labels = left_pad_sequences(flatten_labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            flatten_labels = flatten_input_ids
        assert flatten_labels.shape == flatten_input_ids.shape == flatten_attention_mask.shape, \
            f"{flatten_labels.shape}, {flatten_input_ids.shape}, {flatten_attention_mask.shape}"

        # go through the gpt model as if it's a new batch
        gpt_result = self.gpt(input_ids=flatten_input_ids, attention_mask=flatten_attention_mask, labels=flatten_labels)

        # reorganize the logits to compute individual program log probs
        shift_logits = gpt_result.logits[..., :-1, :].contiguous()
        shift_labels = flatten_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flatten_unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        unreduced_loss = flatten_unreduced_loss.view(shift_labels.shape) * flatten_attention_mask[..., 1:]

        # compute the marginal log prob
        loss = self.get_mixed_mle_mml_loss(unreduced_loss, cached_program_nums)
        self.log("loss", loss, on_step=True, on_epoch=True)
        self.log("fcp_buffer_size", sum(cached_program_nums) / len(cached_program_nums), on_step=False, on_epoch=True)

        return {"loss": loss}
    
    def get_mixed_mle_mml_loss(self, unreduced_loss: torch.Tensor, cached_program_nums: List[int], 
                                log_prob_dist: bool = True) -> torch.Tensor:
        """
        Compute the loss for the MML and MLE.
        """
        # compute the marginal log prob and the sum of the log probs
        grouped_example_log_probs = torch.split(-self.beta_smoothing * torch.sum(unreduced_loss, dim=1), cached_program_nums)
        marginal_log_probs = torch.stack([-1.0 * torch.logsumexp(log_probs, dim=0) / self.beta_smoothing for log_probs in grouped_example_log_probs])
        norm_func = (lambda x: 1.0 ) if not self.mle_aug_norm else (lambda x: 1.0 / len(x))
        sum_log_probs = torch.stack([-norm_func(log_probs) * torch.sum(log_probs, dim=0) for log_probs in grouped_example_log_probs])
        loss = torch.mean(self.mml_lambda * marginal_log_probs + self.mle_lambda * sum_log_probs)

        if log_prob_dist:
            # some additional metrics to evaluate the distribution of the programs
            max_prob = [sorted(torch.exp(log_probs), reverse=True)[0] for log_probs in grouped_example_log_probs]
            second_max_prob = [sorted(torch.exp(log_probs), reverse=True)[1]
                                    if len(log_probs) > 1 else None for log_probs in grouped_example_log_probs]
            second_max_prob = list(filter(lambda x: x is not None, second_max_prob))

            max_prob_avg = float(torch.pow(torch.stack(max_prob).mean(), 1.0 / self.beta_smoothing))
            second_max_prob_avg = float(torch.pow(torch.stack(second_max_prob).mean(), 1.0 / self.beta_smoothing)) \
                                        if len(second_max_prob) > 0 else 0.0

            self.log("max_prob", max_prob_avg, on_step=False, on_epoch=True)
            self.log("second_max_prob", second_max_prob_avg, on_step=False, on_epoch=True)

        return loss

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return super(GptStmtStateModel, self).forward(input_ids, attention_mask, metadata)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        # first do all the things that the base model do
        super().validation_epoch_end(outputs)

        # then save the buffer status
        save_buffer_file_path = os.path.join(self.trainer.log_dir,
                                f'buffer_step_{self.trainer.global_step}_rank_{self.trainer.global_rank}.jsonl')
        with open(save_buffer_file_path, 'w+') as f:
            for task_id, program_seqs in self.correct_program_buffer.items():
                json_dict = {"task_id": task_id, "saved_programs": list(program_seqs)}
                f.write(json.dumps(json_dict) + '\n')
        print(f"buffer saved to {save_buffer_file_path}")

    def training_epoch_end(self, outputs) -> None:
        if not torch.distributed.is_initialized():
            print("training_epoch_end: not using distributed training")
            return

        # gather all the buffers from all processes
        world_size = torch.distributed.get_world_size()
        all_buffer_list = [{} for _ in range(world_size)]
        torch.distributed.all_gather_object(all_buffer_list, self.correct_program_buffer)

        # merge all the buffers
        prev_avg_buffer_size = sum(map(lambda x: len(x[1]), self.correct_program_buffer.items())) / len(self.correct_program_buffer)
        merged_buffer: Dict[str, Set[str]] = {}
        for buffer in all_buffer_list:
            for task_id, programs in buffer.items():
                if task_id not in merged_buffer:
                    merged_buffer[task_id] = programs
                else:
                    merged_buffer[task_id].update(programs)

        self.correct_program_buffer = merged_buffer
        after_avg_buffer_size = sum(map(lambda x: len(x[1]), self.correct_program_buffer.items())) / len(self.correct_program_buffer)
        print(f"buffer size increased from {prev_avg_buffer_size} to {after_avg_buffer_size}, by {after_avg_buffer_size - prev_avg_buffer_size}")

