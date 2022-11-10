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

from .gpt_stmt_mml_model import GptStmtMmlModel
from .gpt_stmt_state_model import GptStmtStateModel
from .gpt_seq2seq_model import post_process_code
from .gpt_util import left_pad_sequences
from execution.execution_evaluation import execution_acc, mathqa_execution, batch_exec_programs, mathqa_answer_eq
from execution.program_tracing import get_execution_states, batch_program_tracing
from execution.program_tracing import ProgState, ProgTrace, ProgTraceUnit, HashableProgState, Program

def get_hashable_state(state_dict: ProgState) -> HashableProgState:
    if len(state_dict) == 0:
        raise ValueError("state_dict is empty, check if the code is empty or being gap")

    # get the values and sort it to make sure the order is the same FIXME: This only works for mathqa
    str_vars = [str(value) for _, value in state_dict.items()]

    # make it immutable to be hashable so that we can use it as a key
    return tuple(sorted(str_vars))

def mathqa_identify_fully_correct_state(state: ProgState, gold_answer: Any):
    if "type" in state and "code" in state:
        raise ValueError("mathqa_identify_fully_correct_state function only accepts states, not tracing units")

    if "answer" not in state:
        return False
    else:
        return mathqa_answer_eq(state["answer"], gold_answer)

def mathqa_identify_output(state: ProgState):
    if "type" in state and "code" in state:
        raise ValueError("mathqa_identify_output function only accepts states, not tracing units")

    if "answer" in state:
        return True
    else:
        return False

def construct_program_from_trace(trace: ProgTrace) -> Program:
    code = "".join([unit.code for unit in trace])
    code_lite = post_process_code(code)
    return Program(code, code_lite, trace)

def get_program_str_set(programs: List[Program]) -> Set[str]:
    return set([program.code_lite for program in programs])

def not_as_prefix_of_programs(program: Program, programs: List[Program]) -> bool:
    for prog in programs:
        if prog.code_lite.startswith(program.code_lite):
            return False
    return True

def get_empty_program() -> Program:
    return Program("", "", [])

def get_num_stmts(program: Program) -> int:
    return len(list(filter(lambda x: x.type == "stmt", program.trace)))

class GptStmtPartialMmlModel(GptStmtMmlModel):
    def __init__(self, 
                 transformer_model_name: str,
                 n_pcp_samples: int = 1,
                 prioritize_fcp: bool = True,
                 length_diff_tolerance: int = 100,
                 sampling_from_states: bool = False,
                 sampling_full_prog_only: bool = False,
                 gold_program_only: bool = False,
                 fcp_only: bool = False,
                 norm_marg_by_len: bool = False,
                 **kwargs) -> None:

        super().__init__(transformer_model_name, **kwargs)

        self.n_pcp_samples = n_pcp_samples
        assert n_pcp_samples == 1, "currently only support n_pcp_samples = 1"
        self.prioritize_fcp = prioritize_fcp
        self.length_diff_tolerance = length_diff_tolerance
        self.sampling_from_states = sampling_from_states
        self.sampling_full_prog_only = sampling_full_prog_only
        self.gold_program_only = gold_program_only
        self.fcp_only = fcp_only
        self.norm_marg_by_len = norm_marg_by_len

        # for each task id as the key, the value dict maps states to known sub-programs
        self.state_programs_dict: Dict[str, Dict[HashableProgState, List[Program]]] = {}

        # redefine it here to use new type hints
        self.correct_program_buffer: Dict[str, List[Program]] = {} 

        # this needs to be differentiated from the fully correct programs 
        # because we do not sample for the fully correct programs
        self.partially_correct_program_buffer: Dict[str, List[Program]] = {} 


    def save_program_by_trace(self, task_id: str, tracing_states: Union[ProgTrace, None],
                              is_fully_correct: bool) -> str:
        """ all the traces of the programs will have the final state being correct, 
            so we try to save all of its subprograms by state

            return: the status of the saving attempt, either one of 
                ["saved", "existing fcp", "existing pcp", "not valid"] """

        # nothing to save if the program executes to error
        if tracing_states is None:
            return "S3: not valid"

        # construct the program tuple
        trace_program = construct_program_from_trace(tracing_states)

        def check_buffer_and_remove(buffer: Dict[str, List[Program]], program: Program):
            # check any of the existing program is the prefix of this new program, if so, remove the shorter one
            programs_to_remove = []
            for i, saved_program in enumerate(buffer[task_id]): 
                # empty program doesn't need to be removed
                if len(saved_program.code_lite) > 0 and program.code_lite.startswith(saved_program.code_lite):
                    programs_to_remove.insert(0, i)
            for idx in programs_to_remove:
                buffer[task_id].pop(idx)

        # we only save the longest program 
        if is_fully_correct:
            if trace_program.code_lite not in get_program_str_set(self.correct_program_buffer[task_id]):
                # check if any partial program is the prefix of this new fully correct program
                check_buffer_and_remove(self.partially_correct_program_buffer, trace_program)
                self.correct_program_buffer[task_id].append(trace_program)
            else:
                # there is nothing to save for the states dict since the full program is already in the buffer
                return "S4: existing fcp"
        elif not_as_prefix_of_programs(trace_program, self.partially_correct_program_buffer[task_id]) and \
             not_as_prefix_of_programs(trace_program, self.correct_program_buffer[task_id]):
            # check if any partial program is the prefix of this new partially correct program
            check_buffer_and_remove(self.partially_correct_program_buffer, trace_program)
            self.partially_correct_program_buffer[task_id].append(trace_program)
        else:
            return "S5: existing pcp"

        # we try to save all the sub-programs of the program by state, excluding the final correct state
        tracing_states_to_save = tracing_states[:-1] if is_fully_correct else tracing_states
        for i, trace_unit in enumerate(tracing_states_to_save):
            if trace_unit.type != "stmt":
                continue

            sub_program = construct_program_from_trace(tracing_states_to_save[:i+1])

            # check the state
            hashable_state = get_hashable_state(trace_unit.state)
            if hashable_state not in self.state_programs_dict[task_id]:
                self.state_programs_dict[task_id][hashable_state] = [sub_program]
            else:
                if sub_program.code_lite not in get_program_str_set(self.state_programs_dict[task_id][hashable_state]): 
                    self.state_programs_dict[task_id][hashable_state].append(sub_program)

        return "S6: saved"

    def check_and_save_partially_correct_program(self, task_id: str, 
                                                 program_trace: Union[ProgTrace, None],
                                                 gold_answer: Any) -> str:
        """ check if there is a sub-program that worth saving """

        if program_trace is None:
            return "S0: not executable"

        # see if the any of the states matches the saved correct states
        saved_states: Dict[HashableProgState, List[Program]] = self.state_programs_dict.get(task_id) 

        # identify the longest partially (or fully) correct program that fits the constraints
        furthest_correct_state_idx = -1
        stmt_num = 0
        for i, tracing_unit in enumerate(program_trace):
            if tracing_unit.type != "stmt":
                continue
            else:
                stmt_num += 1

            if len(tracing_unit.state) == 0:
                continue # most likely because a comment is generated as the first line
            else:
                hashable_state = get_hashable_state(tracing_unit.state)

            reach_full_correct_state = mathqa_identify_fully_correct_state(tracing_unit.state, gold_answer)
            reach_output = mathqa_identify_output(tracing_unit.state)
            if reach_full_correct_state:
                min_fcp_len = min(list(map(get_num_stmts, self.correct_program_buffer[task_id])))
                if stmt_num - min_fcp_len <= self.length_diff_tolerance:
                    furthest_correct_state_idx = i
                    break
                else:
                    return "S1: fully correct but length exceeds tolerance"
            elif reach_output and not reach_full_correct_state:
                # if the program produce the incorrect output, we don't need to save it (but the prefix might still be useful)
                break
            elif hashable_state in saved_states:
                min_pcp_len = min(list(map(get_num_stmts, saved_states[hashable_state])))
                if stmt_num - min_pcp_len <= self.length_diff_tolerance:
                    furthest_correct_state_idx = i

        # save the all the sub-programs of this program by state
        if furthest_correct_state_idx != -1:
            is_fully_correct = mathqa_identify_fully_correct_state(program_trace[furthest_correct_state_idx].state, gold_answer)
            return self.save_program_by_trace(task_id, program_trace[:furthest_correct_state_idx + 1], is_fully_correct)
        else:
            return "S2: not partiallly correct or partially correct but length exceeds tolerance"

    def concat_context_with_multiple_programs(self, context_input_ids: torch.Tensor, 
                                              context_attention_mask: torch.Tensor, 
                                              programs: List[List[Program]],
                                              is_fully_correct: List[List[bool]]) -> str:
        """ concatenate each context tensor with multiple programs """

        # remove the left paddings first and concat the context and cached programs
        context_seqs = [context_input_ids[i, -context_len:] 
                            for i, context_len in enumerate(context_attention_mask.sum(dim=1))]
        cached_program_seqs: List[List[torch.Tensor]] = [[self.tokenizer(prog.code, return_tensors="pt")['input_ids'][0]
                                                            for prog in task_programs] 
                                                                for task_programs in programs]
        flatten_input_ids = []
        for i, program_seqs in enumerate(cached_program_seqs):
            for j, program_seq in enumerate(program_seqs):
                concat_context_program = torch.cat([context_seqs[i], program_seq.to(dtype=context_seqs[i].dtype, device=self.device)], dim=0)
                if is_fully_correct[i][j]:
                    concat_context_program = torch.cat((concat_context_program, torch.tensor([self.tokenizer.eos_token_id], device=self.device)), dim=0)
                flatten_input_ids.append(concat_context_program)
        flatten_attention_mask = [torch.ones_like(flatten_ids) for flatten_ids in flatten_input_ids]

        flatten_input_ids = left_pad_sequences(flatten_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        flatten_attention_mask = left_pad_sequences(flatten_attention_mask, batch_first=True, padding_value=0)

        # exclude the loss from context by setting the labels of them to be -100
        if self.exclude_context_loss:
            flatten_labels = []
            for i, program_seqs in enumerate(cached_program_seqs):
                for j, program_seq in enumerate(program_seqs):
                    concat_labels = torch.cat([-100 * torch.ones_like(context_seqs[i]), 
                                                program_seq.to(dtype=context_seqs[i].dtype, device=self.device)], dim=0)
                    if is_fully_correct[i][j]:
                        concat_labels = torch.cat((concat_labels, torch.tensor([self.tokenizer.eos_token_id], device=self.device)), dim=0)
                    flatten_labels.append(concat_labels)
            flatten_labels = left_pad_sequences(flatten_labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            flatten_labels = flatten_input_ids

        assert flatten_input_ids.shape[0] == len(flatten_attention_mask) == sum([len(progs) for progs in programs]) == flatten_labels.shape[0]

        return flatten_input_ids, flatten_attention_mask, flatten_labels

    def get_marg_program_set(self, task_id: str):
        """ get the programs to marginalize over for each of the task according to the specific settings """

        # TODO: (maybe with fancier policy) first fill in the fully correct programs, 
        # if not enough, fill in the partially correct programs
        programs, is_fully_correct = [], []
        programs.extend(self.correct_program_buffer[task_id])
        is_fully_correct.extend([True] * len(self.correct_program_buffer[task_id]))

        if self.gold_program_only:
            return programs[:1], is_fully_correct[:1]
        if self.fcp_only:
            return programs[:self.marg_set_size], is_fully_correct[:self.marg_set_size]

        non_empty_pcp_buffer = list(filter(lambda x: len(x.code_lite) > 0, self.partially_correct_program_buffer[task_id]))
        programs.extend(non_empty_pcp_buffer)
        is_fully_correct.extend([False] * len(non_empty_pcp_buffer))

        if len(programs) > self.marg_set_size:
            if self.prioritize_fcp:
                return programs[:self.marg_set_size], is_fully_correct[:self.marg_set_size]
            else:
                # random sample the program indices, regardless of being pcp or fcp
                idx_sample = random.sample(range(len(programs)), self.marg_set_size)
                return [programs[i] for i in idx_sample], [is_fully_correct[i] for i in idx_sample]
        else:
            return programs, is_fully_correct

    def get_samples_for_completion(self, task_ids) -> List[List[Program]]:
        """ sample some unfinished programs from the buffer to do the completion sampling """
        if self.sampling_from_states:
            # first sample a state for which the programs reach
            states = [random.sample(self.state_programs_dict[task_id].keys(), 1)[0] for task_id in task_ids]

            # then we sample a program that reaches the state
            programs = [random.sample(self.state_programs_dict[task_id][state], self.n_pcp_samples) 
                            for task_id, state in zip(task_ids, states)]
        elif self.sampling_full_prog_only:
            # get the blank program to sample the full program
            programs = [[self.partially_correct_program_buffer[task_id][0]] for task_id in task_ids]
        else:
            # sample a program from the pcp buffera
            programs = [random.sample(self.partially_correct_program_buffer[task_id], self.n_pcp_samples) 
                            for task_id in task_ids]

        return programs


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # use task ids as the identifier
        task_ids = [ex["task_id"] for ex in batch["metadata"]]

        # add the gold programs to the correct program buffer and empty program to the partially correct program buffer
        for i, example in enumerate(batch["metadata"]):
            task_id = task_ids[i]
            # need to initialize the three data structures with this order
            if task_id not in self.state_programs_dict:
                # in order to be able to sample the empty program when sampling programs to be complete
                self.state_programs_dict[task_id] = {"NULL": [get_empty_program()]}

            if task_id not in self.partially_correct_program_buffer:
                # add empty program to the partially correct program buffer
                self.partially_correct_program_buffer[task_id] = [get_empty_program()]

            if task_id not in self.correct_program_buffer:
                # add the gold programs to the correct program buffer
                self.correct_program_buffer[task_id] = []
                tracing_states = get_execution_states(example["code"])
                return_msg = self.save_program_by_trace(task_id, tracing_states, is_fully_correct=True)
                assert return_msg == "S6: saved"
        
        # do on-policy sampling for the current tasks from the program prefixes
        context_input_ids = batch["input_ids"]
        context_attention_mask = batch["attention_mask"]

        if not self.gold_program_only:
            # TODO: maybe with fancier sampling methods, e.g., sampling from in-between the program
            pcp_samples: List[List[Program]] = self.get_samples_for_completion(task_ids)
            pcp_input_ids, pcp_attention_mask, _ = self.concat_context_with_multiple_programs(context_input_ids, 
                                                        context_attention_mask, pcp_samples, 
                                                        is_fully_correct=[[False]*len(progs) for progs in pcp_samples])

            with torch.no_grad():
                # generate the programs and get their execution results
                max_context_len = pcp_input_ids.size(1)
                output_seqs = self.gpt.generate(input_ids=pcp_input_ids, attention_mask=pcp_attention_mask, 
                                                do_sample=True, max_new_tokens=self.max_sampling_len,
                                                num_return_sequences=self.on_policy_sample_num,
                                                temperature=self.on_policy_sample_temp)
                generated_seqs = output_seqs[:, max_context_len:].cpu()
                generated_program_completions = self.tokenizer.batch_decode(generated_seqs, skip_special_tokens=True)
                generated_programs = [pcp_samples[i // self.on_policy_sample_num][0].code + completion # FIXME: only works when self.n_pcp_samples == 1
                                        for i, completion in enumerate(generated_program_completions)]
                program_traces = batch_program_tracing(generated_programs)

                # save the programs with correct results into the buffer if it's not already in there
                results_count_dict = {f"S{i}": 0 for i in range(7)}
                for i, program_trace in enumerate(program_traces):
                    example_idx = i // self.on_policy_sample_num
                    gold_answer = batch["metadata"][example_idx]["answer"]
                    task_id = task_ids[example_idx]
                    result_msg = self.check_and_save_partially_correct_program(task_id, program_trace, gold_answer)
                    results_count_dict[result_msg[:2]] += 1

                for k in results_count_dict.keys():
                    self.log(f"save_msg_{k}", results_count_dict[k] / len(program_traces), on_step=False, on_epoch=True)

        # select the set of programs to marginalize over for each task and tokenize + concatenate with context
        marg_programs: List[List[Program]] = []
        marg_is_fully_correct: List[List[bool]] = []
        marg_program_nums: List[int] = []
        for task_id in task_ids:
            programs, is_fully_correct = self.get_marg_program_set(task_id)
            marg_programs.append(programs)
            marg_is_fully_correct.append(is_fully_correct)
            marg_program_nums.append(len(programs))
        flatten_input_ids, flatten_attention_mask, flatten_labels = self.concat_context_with_multiple_programs(context_input_ids, 
                                                                        context_attention_mask, marg_programs, marg_is_fully_correct)

        # go through the gpt model as if it's a new batch
        gpt_result = self.gpt(input_ids=flatten_input_ids, attention_mask=flatten_attention_mask, labels=flatten_labels)

        # reorganize the logits to compute individual program log probs
        shift_logits = gpt_result.logits[..., :-1, :].contiguous()
        shift_labels = flatten_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flatten_unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        unreduced_loss =flatten_unreduced_loss.view(shift_labels.shape) * flatten_attention_mask[..., 1:]

        if self.norm_marg_by_len:
            # normalizing by length so we can avoid favoring shorter (e.g., partially-correct) programs
            seq_lens = torch.sum(flatten_attention_mask, dim=-1, keepdim=True)
            grouped_seq_len = torch.split(seq_lens, marg_program_nums)
            max_group_len = torch.cat([torch.full_like(seq_len, torch.max(seq_len)) for seq_len in grouped_seq_len], dim=0)
            assert seq_lens.shape == max_group_len.shape
            unreduced_loss = unreduced_loss / seq_lens * max_group_len

        # compute the marginal log prob
        loss = self.get_mixed_mle_mml_loss(unreduced_loss, marg_program_nums)
        self.log("loss", loss, on_step=True, on_epoch=True)

        self.log("marg_size", sum(marg_program_nums) / len(marg_program_nums), on_step=False, on_epoch=True)
        self.log("fcp_buffer_size", sum([len(self.correct_program_buffer[task_id]) for task_id in task_ids]) \
                                        / len(task_ids), on_step=False, on_epoch=True)
        self.log("pcp_buffer_size", sum([len(self.partially_correct_program_buffer[task_id]) for task_id in task_ids]) \
                                        / len(task_ids), on_step=False, on_epoch=True)
        self.log("state_prog_dict_size", sum([len(self.state_programs_dict[task_id]) for task_id in task_ids]) \
                                        / len(task_ids), on_step=False, on_epoch=True)

        return {"loss": loss}

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return super(GptStmtStateModel, self).forward(input_ids, attention_mask, metadata)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        # first do all the things that the base model do
        super(GptStmtMmlModel, self).validation_epoch_end(outputs)

        # then save the buffer status for the fully correct programs and the partially correct programs
        save_buffer_file_path = os.path.join(self.trainer.log_dir,
                                f'buffer_step_{self.trainer.global_step}_rank_{self.trainer.global_rank}.jsonl')
        with open(save_buffer_file_path, 'w+') as f:
            for task_id, programs in self.correct_program_buffer.items():
                json_dict = {"task_id": task_id, 
                             "saved_fcp_programs": list([prog.code for prog in programs]),
                             "saved_pcp_programs": list([prog.code for prog in self.partially_correct_program_buffer[task_id]])}
                f.write(json.dumps(json_dict) + '\n')
        print(f"buffer saved to {save_buffer_file_path}")

    def merge_buffers(self, is_fcp_buffer: bool):
        # first identify which buffer to merge (full or partial)
        buffer_to_merge = self.correct_program_buffer if is_fcp_buffer else self.partially_correct_program_buffer

        world_size = torch.distributed.get_world_size()
        all_buffer_list: List[Dict[str, List[Program]]] = [{} for _ in range(world_size)]
        torch.distributed.all_gather_object(all_buffer_list, buffer_to_merge)

        # merge all the buffers
        prev_avg_buffer_size = sum(map(lambda x: len(x[1]), buffer_to_merge.items())) / len(buffer_to_merge)

        for buffer in all_buffer_list:
            for task_id, programs in buffer.items():
                if task_id not in buffer_to_merge:
                    assert task_id not in self.correct_program_buffer and \
                        task_id not in self.partially_correct_program_buffer and \
                        task_id not in self.state_programs_dict, \
                        f"task_id {task_id} should not be in any buffer"

                    # init all three data structures
                    # TODO: this can be optimized by simply assigning the three data structures since they are empty
                    self.state_programs_dict[task_id] = {"NULL": [get_empty_program()]}
                    self.partially_correct_program_buffer[task_id] = [get_empty_program()]
                    self.correct_program_buffer[task_id] = []

                for program in programs:
                    self.save_program_by_trace(task_id, program.trace, is_fully_correct=is_fcp_buffer)

        after_avg_buffer_size = sum(map(lambda x: len(x[1]), buffer_to_merge.items())) / len(buffer_to_merge)
        print(f"{'fcp' if is_fcp_buffer else 'pcp'} buffer size increased " \
                    f"from {prev_avg_buffer_size} to {after_avg_buffer_size}, " \
                    f"by {after_avg_buffer_size - prev_avg_buffer_size}")


    def training_epoch_end(self, outputs) -> None:
        if not torch.distributed.is_initialized():
            print("training_epoch_end: not using distributed training")
            return

        self.merge_buffers(is_fcp_buffer=True)
        self.merge_buffers(is_fcp_buffer=False)

