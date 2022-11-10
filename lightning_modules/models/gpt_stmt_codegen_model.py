import torch
import json
import os
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional, Dict, Any, Tuple, List
from overrides import overrides
from nltk import edit_distance
from torchmetrics.functional import rouge_score


from torchmetrics import BLEUScore, ROUGEScore, MeanMetric
from pytorch_lightning import LightningModule
from transformers.optimization import AdamW

from .gpt_util import get_gpt
from lightning_modules.datasets.reader_utils import END_OF_CELL_TOKEN

class GptStmtCodeGenModel(LightningModule):
    def __init__(self, transformer_model_name: str, 
                 prediction_file_name: str = None, 
                 max_stmt_len: int = 100, 
                 max_stmt_num: int = 30,
                 beam_size: int = 1,
                 max_context_len: int = 1024,
                 relatexed_em: bool = True,
                 max_ref_num: int = 5,
                 opt_lr: float = 1e-4) -> None:
        super().__init__()
        self._model_name = transformer_model_name
        self.max_stmt_len = max_stmt_len
        self.max_stmt_num = max_stmt_num
        self.beam_size = beam_size
        self.max_context_len = max_context_len
        self.relaxed_em = relatexed_em
        self.max_ref_num = max_ref_num

        self.opt_lr = opt_lr

        self.prediction_file_name = prediction_file_name
        if self.prediction_file_name:
            self.output_file_path = open(f"{prediction_file_name}_output.jsonl", "w")

        # We only instantiate this when we need it.
        self.gpt, self.tokenizer = get_gpt(transformer_model_name, 
                                           additional_special_tokens=[END_OF_CELL_TOKEN])

        # save predictions for every validation epoch
        self.predictions = []

        self._rouge_metric = ROUGEScore()
        self._bleu_metric = BLEUScore()
        self._em_metric = MeanMetric()
        self._stmt_length = MeanMetric()
        self._cell_stmt_num = MeanMetric()
        self._edit_distance = MeanMetric()
        self._perplexity = MeanMetric()
        self._val_loss = MeanMetric()


    def forward(  # type: ignore
        self, 
        context_tokens: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        input_tokens: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        The inference time behavior of the model.

        Args:
            context_tokens (Optional[torch.Tensor], optional): Tokens from the context. Defaults to None.
            context_mask (Optional[torch.Tensor], optional): Mask out all non-context (including target) tokens. Defaults to None.
            input_tokens (Optional[torch.Tensor], optional): Concatenated context and target tokens. Defaults to None.
            input_mask (Optional[torch.Tensor], optional): Mask out all non-input (including padding) tokens. Defaults to None.
            target_mask (Optional[torch.Tensor], optional): Mask out all non-target tokens, used to correctly calculate \
                the loss. Defaults to None.
            metadata (Optional[List[Dict[str, Any]]], optional): All additional information, `List` for the batch. Defaults to None.

        Returns:
            Dict[str, Any]: results saved in a `Dict` object.
        """        

        assert all(['target_str' in d for d in metadata]), "All the batch must have a `target_str` field in metadata"

        # calculate the perplexity of the model
        input_ids = input_tokens 
        attention_mask = input_mask

        # at inference time, since we do not skip instances, the target may be too long to calculate the loss
        # in this case, we truncate the input_ids and attention_mask to the max length that GPT-Neo can accept: 2048
        if input_ids.shape[1] > self.gpt.config.max_position_embeddings:
            assert not self.training
            input_ids = input_ids[:, :self.gpt.config.max_position_embeddings]
            attention_mask = attention_mask[:, :self.gpt.config.max_position_embeddings]
            target_mask = target_mask[:, :self.gpt.config.max_position_embeddings]


        labels = torch.clone(input_ids).masked_fill(~target_mask, -100)
        gpt_result = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # step_log_probs = F.log_softmax(gpt_result.logits, dim=2)
        # gold_vocab_log_probs = torch.gather(step_log_probs, 2, input_ids.unsqueeze(2))
        # masked_target_log_probs = gold_vocab_log_probs.squeeze(2) * target_mask.float()
        # perplexity = torch.exp(- torch.sum(masked_target_log_probs, dim=1) / torch.sum(target_mask, dim=1))

        batch_size = context_tokens.shape[0]
        input_ids, attention_mask = context_tokens, context_mask

        loss = float(gpt_result.loss)
        perplexity = float(torch.exp(gpt_result.loss))
        
        output_dicts = [{"generated_stmt_lens": [], 
                         "perplexity": perplexity, 
                         "val_loss": loss} 
                        for i in range(batch_size)]

        # iterative stmt generation
        completed_cells = [[] for _ in range(batch_size)]
        incomplete_cell_indices = list(range(batch_size))
        for stmt_idx in range(self.max_stmt_num):
            # where attention mask is 0, the token id must be self.tokenizer.pad_token_id TODO: this is only for debugging
            inner_batch_size = len(input_ids)
            # assert all([all([attention_mask[i][j] == 1 or input_ids[i][j] == self.tokenizer.pad_token_id \
            #                     for j in range(len(input_ids[i]))]) for i in range(inner_batch_size)])

            max_context_len = input_ids.size(1)
            context_len_list = attention_mask.sum(dim=1) 
            output_seqs = self.gpt.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                            do_sample=False, max_length=self.max_stmt_len+max_context_len,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            num_beams=self.beam_size, num_return_sequences=self.beam_size)

            # remove the context and the tokens after the first eos token in the generated seq
            generated_seqs = [output_seqs[:, max_context_len:][i] for i in range(inner_batch_size)]
            for i, output_seq in enumerate(generated_seqs):
                eos_indices = (output_seq == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_eos_idx = int(eos_indices[0])
                    generated_seqs[i] = output_seq[:first_eos_idx+1]
                else:
                    # this means that the generation hits the max_stmt_len, so we need to manually add <eos>
                    generated_seqs[i] = torch.cat([output_seq, torch.tensor([self.tokenizer.eos_token_id], device=output_seq.device)])
                output_dicts[incomplete_cell_indices[i]]["generated_stmt_lens"].append(len(generated_seqs[i]))

            # concat the context with the generated seqs
            full_seqs = []
            for i in range(inner_batch_size):
                full_seq = torch.cat([input_ids[i][:context_len_list[i]], generated_seqs[i]])
                full_seqs.append(full_seq)

            # check if the end_of_cell token is in the generated output
            incomplete_output_seqs = []
            incomplete_cell_indices_new = []
            for i, output_seq in enumerate(full_seqs):
                eoc_indices = (output_seq == self.tokenizer.convert_tokens_to_ids(END_OF_CELL_TOKEN)) \
                                                                .nonzero(as_tuple=True)[0]
                # filter out those eoc tokens in the context
                eoc_indices = list(filter(lambda x: x >= context_len_list[i], eoc_indices))
                if len(eoc_indices) > 0: # eoc detected, end the stmt generation loop
                    # cut off **after** the [eoc, eos] tokens
                    first_eoc_idx = int(eoc_indices[0])
                    completed_cells[incomplete_cell_indices[i]].extend(output_seq[context_len_list[i]:first_eoc_idx+2])
                    output_dicts[incomplete_cell_indices[i]]['stmt_num'] = stmt_idx
                else:
                    incomplete_output_seqs.append(output_seq)
                    completed_cells[incomplete_cell_indices[i]].extend(output_seq[context_len_list[i]:])
                    incomplete_cell_indices_new.append(incomplete_cell_indices[i])
            incomplete_cell_indices = incomplete_cell_indices_new

            if len(incomplete_output_seqs) == 0:
                # all seqs have been completed by generating the eoc token
                break
            elif stmt_idx == self.max_stmt_num - 1:
                # reach the max stmt num, but still not all the seqs are completed
                for i, incomplete_cell in enumerate(incomplete_output_seqs):
                    output_dicts[incomplete_cell_indices[i]]['stmt_num'] = stmt_idx
                    completed_cells[incomplete_cell_indices[i]].extend(
                        torch.tensor([self.tokenizer.additional_special_tokens_ids[0], self.tokenizer.eos_token_id], 
                                                                    device=output_seq.device))
                break

            # reformulate the input_ids and attention_mask with the newly generated output
            incomplete_output_seqs = [output_seq[-self.max_context_len:] for output_seq in incomplete_output_seqs]
            attention_mask_list = [torch.ones_like(incomplete_output_seq) for incomplete_output_seq in incomplete_output_seqs]
            # pad to the same length and stack to be the new input_ids
            input_ids = torch.nn.utils.rnn.pad_sequence(incomplete_output_seqs, batch_first=True, 
                                                        padding_value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=False)

        # completed_cells are accumlated stmts for each cell, not including the original context; decode back to the strs
        cell_strs = self.tokenizer.batch_decode(completed_cells)

        # save the results and all metadata in the output dict
        for i in range(batch_size):
            output_dicts[i]['predicted'] = cell_strs[i]
            output_dicts[i].update(metadata[i])

        return output_dicts

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_tokens']
        attention_mask = batch['input_mask']
        target_mask = batch['target_mask']


        # mask the context tokens when computing the loss (by default, hf won't compute loss for the token_id -100)
        labels = torch.clone(input_ids).masked_fill(~target_mask, -100)
        gpt_result = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.log("loss", gpt_result.loss, on_step=True, on_epoch=True)
        return {"loss": gpt_result.loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        # input_tokens, target_mask, context_tokens, target_tokens, metadata = batch
        return self.forward(batch['context_tokens'], 
                            batch['context_mask'],
                            batch['input_tokens'],
                            batch['input_mask'],
                            batch['target_mask'],
                            batch['metadata'])

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        # save the outputs to the model
        self.predictions.extend(outputs)

        # update the evaluation metrics
        for output_dict in outputs:
            self._perplexity(output_dict['perplexity'])
            self._val_loss(output_dict['val_loss'])
            self._cell_stmt_num(len(output_dict['generated_stmt_lens']))
            self._stmt_length(sum(output_dict['generated_stmt_lens']) / len(output_dict['generated_stmt_lens']))

            predicted_cell_str = output_dict['predicted'].replace(self.tokenizer.eos_token, '').replace(END_OF_CELL_TOKEN, '')

            # adapt for both single target and multiple targets evaluation
            if isinstance(output_dict['target_str'], str):
                gold_cell_strs = [output_dict['target_str']]
            else:
                gold_cell_strs = output_dict['target_str']
            gold_cell_strs = [cstr.replace(self.tokenizer.eos_token, '').replace(END_OF_CELL_TOKEN, '') for cstr in gold_cell_strs]

            rouge_scores, em_scores, edit_dists = [], [], []
            em_func = lambda x, y: x.strip() == y.strip() if self.relaxed_em else x == y
            for i, gold_cell_str in enumerate(gold_cell_strs[:self.max_ref_num]):
                rouge_scores.append((rouge_score(predicted_cell_str, gold_cell_str)['rougeLsum_fmeasure'], i))
                em_scores.append((em_func(predicted_cell_str, gold_cell_str), i))
                edit_dists.append((edit_distance(predicted_cell_str, gold_cell_str), i))

            # load the best scores from multiple referenced targets
            best_rouge_idx = max(rouge_scores, key=lambda x: x[0])[1]
            self._rouge_metric(predicted_cell_str, gold_cell_str[best_rouge_idx])
            # self._bleu_metric([[gold_cell_str]], [predicted_cell_str]) # bleu in torchmetrics is just weird
            self._em_metric(max(em_scores, key=lambda x: x[0])[0])
            self._edit_distance(max(edit_dists, key=lambda x: x[0])[0])

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        # log and save the evalution metrics
        self.log_save_metrics()

        # save the predictions
        save_pred_file_path = os.path.join(self.trainer.log_dir,
                                f'predictions_step_{self.trainer.global_step}_rank_{self.trainer.global_rank}.jsonl')
        with open(save_pred_file_path, 'w+') as f:
            for prediction in self.predictions:
                f.write(json.dumps(prediction)+'\n')
        print(f"{len(self.predictions)} predictions saved to {save_pred_file_path}")
        self.predictions = []

    def log_save_metrics(self) -> None:
        # calculate the validation metrics and save to logs as well as files
        metrics = {}

        rouge_dict = dict([(k, float(v)) for k, v in self._rouge_metric.compute().items() if k.endswith('fmeasure')])
        metrics.update(rouge_dict)
        metrics["bleu"] = float(self._bleu_metric.compute())
        metrics["cell_exact_match"] = float(self._em_metric.compute())
        metrics["output_stmt_len"] = float(self._stmt_length.compute())
        metrics["output_stmt_num"] = float(self._cell_stmt_num.compute())
        metrics["cell_edit_dist"] = float(self._edit_distance.compute())
        metrics["perplexity"] = float(self._perplexity.compute())
        metrics["val_loss"] = float(self._val_loss.compute())
        self.log_dict(metrics)

        # NOTE: seems like things as is_global_zero or self.rank == 0 will cause a hang
        # # save the evaluation metrics
        # if self.trainer.is_global_zero:
        #     save_metrics_file_path = os.path.join(self.trainer.log_dir,
        #                                 f'metrics_step_{self.trainer.global_step}.json')
        #     with open(save_metrics_file_path, 'w+') as f:
        #         f.write(json.dumps(metrics, indent=4))
        #     print(f"Eval metrics saved to {save_metrics_file_path}")

        self._rouge_metric.reset()
        self._bleu_metric.reset()
        self._em_metric.reset()
        self._stmt_length.reset()
        self._cell_stmt_num.reset()
        self._edit_distance.reset()
        self._perplexity.reset()
        self._val_loss.reset()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.opt_lr, betas = (0.9, 0.95), eps=1e-8, weight_decay=0.1)
    #     return optimizer
