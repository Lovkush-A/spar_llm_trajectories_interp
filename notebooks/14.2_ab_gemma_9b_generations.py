import circuitsvis as cv
import numpy as np
import torch
from taker import Model
from taker.hooks import HookConfig
from datetime import datetime
import json
from os import listdir
from os.path import exists

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import csv

m = Model("google/gemma-2-9b-it")
m.show_details()

with open('../promptsV1.csv', newline='') as f:
    reader = csv.reader(f)
    readdata = list(reader)
    readdata = readdata[:20]
    print(len(readdata))

import sys, os
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

filename = f"../gemma9b_results/latest_orig_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

max_new_tokens = 200
temperature = 0.3

[h.reset() for h in m.hooks.neuron_replace.values()] #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
for prompt in readdata:
    prompt = prompt[0]
    with HiddenPrints():
        for i in range(50):
            output = m.generate(prompt, max_new_tokens, temperature=temperature)
            
            data = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "model": m.model_repo,
                "type": "original",
                "transplant_layers": None,
                "prompt": prompt,
                "output": output[1],
            }

            with open(filename, "a") as file:
                file.write(json.dumps(data) + "\n")

filename = f"../gemma9b_results/latest_neutral_generation.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass

neutral_prompts = ["\n\n"]
for neutral in neutral_prompts:                
    with HiddenPrints():
        for i in range(50):
            output = m.generate(neutral, max_new_tokens, temperature=temperature)
            
            data = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "model": m.model_repo,
                "type": "neutral",
                "transplant_layers": None,
                "prompt": neutral,
                "output": output[1],
            }

            with open(filename, "a") as file:
                file.write(json.dumps(data) + "\n")


orig_df = pd.read_json(f"../gemma_results/latest_orig_generation.jsonl", lines=True)
def split_at_double_newline(text):
    # Ensure we are only working with strings longer than 15 characters
    if len(text) > 15:
        # Search for the first double newline after the 15th character
        pos = text.find('\n\n', 15)
        if pos != -1:  # Check if double newline was found
            return text[:pos], text[pos:]  # Split and remove the newline from the second part
    return text, None  # If no split is required, return the original text and None

# Apply the function to the DataFrame column
orig_df['paragraph1'], orig_df['paragraph2'] = zip(*orig_df['output'].apply(split_at_double_newline))
orig_df['paragraph1'] = orig_df['prompt'].astype(str) + orig_df['paragraph1'].astype(str)
print(orig_df.head())

filename = f"../gemma_results/latest_transferred_generation_2token.jsonl"
if not exists(filename):
    with open(filename, "w") as f:
        pass
    
for info_prompt in orig_df['paragraph1'][912:]:
    acts = m.get_midlayer_activations(info_prompt)
    orig_token_index = m.get_ids(info_prompt).shape[1] - 1
    for neutral_prompt in neutral_prompts:
        new_token_index  = m.get_ids(neutral_prompt).shape[1] - 1        

        [h.reset() for h in m.hooks.neuron_replace.values()] #RESET HOOKS BEFORE TRANSPLANTING NEXT SET OF ACTIVATIONS
        for layer_index in range(0,26):
            m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index, acts["mlp"][0, layer_index, orig_token_index])
            m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index, acts["attn"][0, layer_index, orig_token_index])
            m.hooks.neuron_replace[f"layer_{layer_index}_mlp_pre_out"].add_token(new_token_index-1, acts["mlp"][0, layer_index, orig_token_index-1])
            m.hooks.neuron_replace[f"layer_{layer_index}_attn_pre_out"].add_token(new_token_index-1, acts["attn"][0, layer_index, orig_token_index-1])
        with HiddenPrints():
            # for i in range(1):
                output = m.generate(neutral_prompt, max_new_tokens, temperature=temperature)
                
                data = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "model": m.model_repo,
                    "type": "transferred",
                    "num_transferred_tokens": 2,
                    "transplant_layers": (0,26),
                    "orig_prompt": info_prompt,
                    "transplant_prompt": neutral_prompt,
                    "output": output[1],
                }

                with open(filename, "a") as file:
                    file.write(json.dumps(data) + "\n")


