from typing import Dict, Iterable, List, Any, Optional, Union

from execution.program_tracing import get_state_repr, is_trivial_state
from .mathqa_line_reader import MathQADataModule, MathQADataset

class MathQAStateDataset(MathQADataset):
    def __init__(
        self, 
        file_path: str,
        transformer_model_name: str, 
        max_instances: int,
        include_answer_state_in_training: bool = False,
        only_include_line_var: bool = False,
        mask_non_value_state_tokens: bool = False,
        mask_all_state_tokens: bool = False,
        skip_trivial_states: bool = False,
        **kwargs):

        self.include_answer_state_in_training = include_answer_state_in_training
        self.only_include_line_var = only_include_line_var

        assert not (mask_non_value_state_tokens and mask_all_state_tokens), \
            "either mask the non value state tokens or all state tokens, not both"
        self.mask_non_value_state_tokens = mask_non_value_state_tokens
        self.mask_all_state_tokens = mask_all_state_tokens
        self.skip_trivial_states = skip_trivial_states

        super().__init__(file_path, transformer_model_name, max_instances, **kwargs)

    def get_input_with_state(self, example: Dict[str, Any]) -> str:
        states = example["states"]

        # add the states as a comment in the previous line to the next stmt
        input = ""
        current_state_dict = {}
        last_stmt = ""
        last_var = ""
        for json_dict in states:
            if json_dict["type"] == "stmt":
                state_repr = get_state_repr(current_state_dict, only_include_keys=[last_var], prev_stmt=last_stmt, 
                                            skip_trivial_states=self.skip_trivial_states)
                input += state_repr + json_dict["code"]
                current_state_dict = json_dict["execution_state"]
                last_stmt = json_dict["code"]
                last_var = last_stmt.split(" ")[0]
            else:
                input += json_dict["code"]
        
        # answer must be in the final state dict
        assert "answer" in current_state_dict
        if self.include_answer_state_in_training:
            # the answer state is never trivial
            input += "\n" + get_state_repr(current_state_dict, only_include_keys=[last_var], skip_trivial_states=False)

        return input

    def mask_out_state_tokens(self, input_str: str):
        hash_token_id = self.tokenizer.encode("#")[0]
        semicolon_token_id = self.tokenizer.encode(";")[0]
        space_eq_token_id = self.tokenizer.encode(" =")[0]
        nl_token_id = self.tokenizer.encode("\n")[0]

        non_value_state_mask = []
        all_state_mask = []
        tokenizer_outputs = self.tokenizer(input_str)
        input_ids = tokenizer_outputs["input_ids"]

        # skip the textual tokens
        assert input_ids[0] == hash_token_id
        text_end_idx = input_ids.index(nl_token_id)
        ids = input_ids[text_end_idx+1:]
        non_value_state_mask.extend([1] * (text_end_idx+1))
        all_state_mask.extend([1] * (text_end_idx+1))

        enter_comment = False
        enter_value = False
        for id in ids:
            if id == hash_token_id:
                enter_comment = True
                non_value_state_mask.append(0)
                all_state_mask.append(0)
            elif id == space_eq_token_id and enter_comment:
                enter_value = True
                non_value_state_mask.append(0)
                all_state_mask.append(0)
            elif id == semicolon_token_id and enter_value:
                enter_value = False
                non_value_state_mask.append(0)
                all_state_mask.append(0)
            elif id == nl_token_id and enter_comment:
                enter_comment = False
                non_value_state_mask.append(0)
                all_state_mask.append(0)
            else:
                # not special token, need to determine the mask based on the state
                if enter_comment and not enter_value:
                    non_value_state_mask.append(0)
                    all_state_mask.append(0)
                elif enter_comment and enter_value:
                    non_value_state_mask.append(1)
                    all_state_mask.append(0)
                else:
                    non_value_state_mask.append(1)
                    all_state_mask.append(1)

        return non_value_state_mask, all_state_mask, tokenizer_outputs

    def get_train_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        code_input = "\n".join([example["text"], self.get_input_with_state(example)])
        non_value_state_mask, all_state_mask, tokenizer_outputs = self.mask_out_state_tokens(code_input)

        example_dict["input_ids"] = tokenizer_outputs["input_ids"] + [self.tokenizer.eos_token_id]
        example_dict["attention_mask"] = tokenizer_outputs["attention_mask"] + [1]
        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        # state labels and state mask
        non_value_state_mask.append(1)
        all_state_mask.append(1)

        if self.mask_non_value_state_tokens:
            label_mask = non_value_state_mask
        elif self.mask_all_state_tokens:
            label_mask = all_state_mask
        else:
            label_mask = [1] * len(example_dict["input_ids"])

        example_dict["state_mask"] = all_state_mask
        example_dict["labels"] = list(map(lambda x: x[0] * x[1] + -100 * (1 - x[1]), zip(example_dict["input_ids"], label_mask)))

        # deal with potential cutoffs
        if len(example_dict["input_ids"]) > 2048:
            print(f"Cutoff instance of length {len(example_dict['input_ids'])}")
            example_dict["input_ids"] = example_dict["input_ids"][:2048]
            example_dict["attention_mask"] = example_dict["attention_mask"][:2048]
            example_dict["labels"] = example_dict["labels"][:2048]
            example_dict["state_mask"] = example_dict["state_mask"][:2048]

        return example_dict

    def get_test_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        tokenizer_outputs = self.tokenizer(example["text"] + "\n" + get_state_repr({}, skip_trivial_states=self.skip_trivial_states))
                                

        example_dict["input_ids"] = tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = tokenizer_outputs["attention_mask"]
        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict

    def get_test_few_shot_instance(self, example: Dict[str, Any], 
                                   few_shot_text_list: List[str],
                                   few_shot_code_list: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class MathQAStateDataModule(MathQADataModule):
    def __init__(self, 
                transformer_model_name: str,
                include_answer_state_in_training: bool = False,
                mask_non_value_state_tokens: bool = False,
                mask_all_state_tokens: bool = False,
                skip_trivial_states: bool = False,
                **kwargs):
        self.include_answer_state_in_training = include_answer_state_in_training
        self.mask_non_value_state_tokens = mask_non_value_state_tokens
        self.mask_all_state_tokens = mask_all_state_tokens
        self.skip_trivial_states = skip_trivial_states

        super().__init__(transformer_model_name, **kwargs)

    def setup(self, stage: Optional[str] = None):
        assert stage in ["fit", "validate", "test"]

        train_data = MathQAStateDataset(file_path=self.train_file_path,
                                   transformer_model_name=self.transformer_model_name,
                                   max_instances=self.train_max_instances, 
                                   mode="train", few_shot_n=self.few_shot_n,
                                   include_answer_state_in_training=self.include_answer_state_in_training,
                                   mask_non_value_state_tokens=self.mask_non_value_state_tokens,
                                   mask_all_state_tokens=self.mask_all_state_tokens,
                                   skip_trivial_states=self.skip_trivial_states)
        self.train_data = train_data

        val_data = MathQAStateDataset(file_path=self.val_file_path,
                                 transformer_model_name=self.transformer_model_name,
                                 max_instances=self.val_max_instances, 
                                 mode="test", few_shot_n=self.few_shot_n, 
                                 include_answer_state_in_training=self.include_answer_state_in_training,
                                 mask_non_value_state_tokens=self.mask_non_value_state_tokens,
                                 mask_all_state_tokens=self.mask_all_state_tokens,
                                 skip_trivial_states=self.skip_trivial_states)
        self.val_data = val_data 