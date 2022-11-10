from numpy import dtype
import torch
import json
import os
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule

from .gpt_util import get_gpt, sanity_check, left_pad_sequences
from .gpt_seq2seq_model import GptSeq2SeqModel
from execution.execution_evaluation import execution_acc, mathqa_execution
from execution.execution_evaluation import execution_eval_at_k, batch_execution_acc
from execution.program_tracing import exec_stmt_in_context, get_state_repr, is_trivial_state
from execution.safe_execution_util import canonicalize_var_dict

class GptStmtStateModel(GptSeq2SeqModel):
    def __init__(self, 
                 transformer_model_name: str,
                 max_stmt_len: int = 20,
                 max_stmt_num: int = 20,
                 max_context_len: int = 1024,
                 eval_with_states: bool = False,
                 skip_trivial_states: bool = False,
                 **kwargs) -> None:
        super().__init__(transformer_model_name, **kwargs)

        self.max_stmt_len = max_stmt_len
        self.max_stmt_num = max_stmt_num
        self.max_context_len = max_context_len
        self.eval_with_states = eval_with_states
        self.skip_trivial_states = skip_trivial_states

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        The inference time behavior of the model.

        Args:
            input_ids [torch.Tensor]: Tokens from the context. 
            metadata (Optional[List[Dict[str, Any]]], optional): All additional information, `List` for the batch. Defaults to None.

        Returns:
            Dict[str, Any]: results saved in a `Dict` object.
        """        
        output_dicts = [{"metadata": metadata[i]} for i in range(len(metadata))]

        # iteratively generate the stmts until eos is generated
        batch_size = len(metadata)
        completed_programs = [[] for _ in range(batch_size)]
        program_states_list = [[{}] for _ in range(batch_size)]
        incomplete_program_indices = list(range(batch_size))
        for stmt_idx in range(self.max_stmt_num):
            # this is how many examples are left in the batch
            inner_batch_size = len(input_ids)

            max_context_len = input_ids.size(1)
            context_len_list = attention_mask.sum(dim=1) 

            output_seqs = self.gpt.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                            do_sample=False, max_length=self.max_gen_len+max_context_len) # FIXME: this should be self.max_stmt_len
                                            # temperature=self.sampling_temp) # NOTE: now we assume only one seq is returned

            # remove the context and the tokens after the first newline token in the generated seq
            generated_seqs = [output_seqs[:, max_context_len:][i] for i in range(inner_batch_size)]
            for i, output_seq in enumerate(generated_seqs):
                nl_indices = (output_seq == self.tokenizer._convert_token_to_id(self.tokenizer.tokenize("\n")[0])).nonzero(as_tuple=True)[0]
                if len(nl_indices) > 0:
                    first_nl_idx = int(nl_indices[0])
                    generated_seqs[i] = output_seq[:first_nl_idx+1] # +1 because we need to include the newline token
                else:
                    # this means that the generation hits the max_stmt_len before the first newline token or an early
                    # eos token is generated. Either way, we need to stop the generation, so we snap an (possibly 
                    # additional) eos token to the end of the generated seq.
                    generated_seqs[i] = torch.cat([output_seq, torch.tensor([self.tokenizer.eos_token_id], device=output_seq.device)])

            # concat the context with the generated seqs
            # full_seqs = []
            # for i in range(inner_batch_size):
            #     full_seq = torch.cat([input_ids[i][:context_len_list[i]], generated_seqs[i]])
            #     full_seqs.append(full_seq)

            # check if the end_of_sequence token is in the generated output
            incomplete_output_seqs = []
            incomplete_program_indices_new = []
            for i, output_seq in enumerate(generated_seqs):
                eos_indices = (output_seq == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0: # eos detected, end the stmt generation loop
                    # cut off **at** the eos token and it's not added to the incomplete list (it's finished)
                    first_eos_idx = int(eos_indices[0])
                    output_seq = output_seq[:first_eos_idx]
                    completed_programs[incomplete_program_indices[i]].extend(output_seq)

                    # get the last state
                    if self.eval_with_states:
                        stmt_str = self.tokenizer.decode(output_seq, skip_special_tokens=True)
                        last_state_dict = program_states_list[incomplete_program_indices[i]][-1]
                        output_state = exec_stmt_in_context(stmt_str, last_state_dict)
                        program_states_list[incomplete_program_indices[i]].append(output_state)
                else:
                    if self.eval_with_states:
                        # the generation is not finished yet, we need to augment the next steps with state information
                        stmt_str = self.tokenizer.decode(output_seq, skip_special_tokens=True)
                        last_state_dict = program_states_list[incomplete_program_indices[i]][-1]
                        output_state = exec_stmt_in_context(stmt_str, last_state_dict)
                        program_states_list[incomplete_program_indices[i]].append(output_state)

                        if output_state == None:
                            # the program will be not successfully executed, so we remove the program from the incomplete list
                            completed_programs[incomplete_program_indices[i]].extend(output_seq)
                        else:
                            # incorporate the state into the context
                            state_str = get_state_repr(output_state, only_include_keys=[stmt_str.split(" ")[0]], 
                                            prev_stmt=stmt_str, skip_trivial_states=self.skip_trivial_states)
                            state_tensor = self.tokenizer.encode(state_str, add_special_tokens=False, return_tensors="pt")[0] \
                                                        .to(device=output_seq.device, dtype=output_seq.dtype)
                            output_seq = torch.cat([output_seq, state_tensor])

                            incomplete_output_seqs.append(torch.cat([input_ids[i][-context_len_list[i]:], output_seq]))
                            completed_programs[incomplete_program_indices[i]].extend(output_seq)
                            incomplete_program_indices_new.append(incomplete_program_indices[i])
                    else:
                        incomplete_output_seqs.append(torch.cat([input_ids[i][-context_len_list[i]:], output_seq]))
                        completed_programs[incomplete_program_indices[i]].extend(output_seq)
                        incomplete_program_indices_new.append(incomplete_program_indices[i])

            incomplete_program_indices = incomplete_program_indices_new

            if len(incomplete_output_seqs) == 0:
                # all seqs have been completed by generating the eos token
                break
            elif stmt_idx == self.max_stmt_num - 1:
                # reach the max stmt num, but still not all the seqs are completed
                # for i, incomplete_cell in enumerate(incomplete_output_seqs):
                #     completed_programs[incomplete_program_indices[i]].extend(
                #         torch.tensor([self.tokenizer.eos_token_id], device=output_seq.device))
                break

            # reformulate the input_ids and attention_mask with the newly generated output
            incomplete_output_seqs = [output_seq[-self.max_context_len:] for output_seq in incomplete_output_seqs]
            attention_mask_list = [torch.ones_like(incomplete_output_seq) for incomplete_output_seq in incomplete_output_seqs]
            # pad to the same length and stack to be the new input_ids
            input_ids = left_pad_sequences(incomplete_output_seqs, batch_first=True, 
                                                        padding_value=self.tokenizer.eos_token_id)
            attention_mask = left_pad_sequences(attention_mask_list, batch_first=True, padding_value=False)

        # completed_cells are accumlated stmts for each program, not including the original context; decode back to the strs
        generated_programs= self.tokenizer.batch_decode(completed_programs)

        for i in range(len(metadata)):
            output_dicts[i].update({"generated_program": generated_programs[i]})
            output_dicts[i].update({"generated_program_state_list": program_states_list[i]})

        return output_dicts

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"] if "labels" in batch else input_ids

        gpt_result = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log("loss", gpt_result.loss, on_step=True, on_epoch=True)

        if "state_mask" in batch:
            # log separately for loss on state tokens and non-state tokens
            state_mask = batch["state_mask"]

            # Shift so that tokens < n predict n
            shift_logits = gpt_result.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_state_mask = state_mask[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            unreduced_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            code_token_loss = torch.sum(shift_state_mask.view(-1) * unreduced_loss) / torch.sum(shift_state_mask)
            state_token_loss = torch.sum((1 - shift_state_mask.view(-1)) * unreduced_loss) / torch.sum(1 - shift_state_mask)

            self.log("code_token_loss", code_token_loss, on_step=True, on_epoch=True)
            self.log("state_token_loss", state_token_loss, on_step=True, on_epoch=True)

        return {"loss": gpt_result.loss}

    def sanity_check_validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        # update the evaluation metrics
        for output_dict in outputs:
            last_state_dict = output_dict["generated_program_state_list"][-1]
            if last_state_dict is not None and "answer" in last_state_dict:
                exec_rate = 1.0
                if last_state_dict["answer"] == output_dict["metadata"]["answer"]:
                    exec_acc = 1.0
                else:
                    exec_acc = 0.0
            else:
                exec_rate = 0.0
                exec_acc = 0.0
            output_dict.pop("generated_program_state_list")

            program_len = len(list(filter(lambda x: not x.startswith("#"), 
                                                output_dict["generated_program"].split("\n"))))
            gold_program_len = len(list(filter(lambda x: not x.startswith("#"), output_dict["metadata"]["code"].split("\n"))))
            program_len_diff = program_len - gold_program_len

            self._num_metric_1(exec_acc)
            self._num_metric_2(exec_rate)

            self._num_metric_3(program_len_diff)

            output_dict["metrics"] = {"exec_acc": float(exec_acc), 
                                      "exec_rate": float(exec_rate),
                                      "program_len_diff": float(program_len_diff)}

        # save the outputs to the model
        self.predictions.extend(outputs)
