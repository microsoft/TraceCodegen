"""
We can only access OpenAI codex through restful API.
"""

import os
import openai
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import deepspeed

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("zero3.json") as f:
    ds_config = json.load(f)

# print(openai.Engine.list())

def codex_prompting(prompt: str, engine: str = "cushman-codex", temperature: float = 0.1,
                    max_tokens: int = 50, stop: str = "\n", n_completions: int = 1) -> str:
    # TODO: figure out a way to print warnings on the prompt and max generation length based on the models
    # For now, as a reference, "cushman-codex" accepts max 2048 tokens and "davinci-codex" accepts max 4096 tokens
    response = openai.Completion.create(engine=engine, temperature=temperature, n=n_completions,
                                        prompt=prompt, max_tokens=max_tokens, stop=stop)
    return [choice["text"] for choice in response["choices"]]

def gpt_model_prompting(prompt: str, model_name: str = "117M", temperature: float = 0.1,
                        max_tokens: int = 50, stop: str = "\n", n_completions: int = 1) -> str:

    device = torch.device("cuda:0")
    dschf = HfDeepSpeedConfig(ds_config)
    
    if model_name == "EleutherAI/gpt-j-6B":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    
    engine = deepspeed.initialize(model=model, config_params=ds_config)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gen_tokens = engine.generate(input_ids=input_ids, max_length=max_tokens, temperature=temperature)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text


if __name__ == "__main__":
    print(gpt_model_prompting("Hello, how are you doing", model_name="EleutherAI/gpt-j-6B", temperature=0.2, max_tokens=50))
