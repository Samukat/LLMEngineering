from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import GenerationConfig
from collections import defaultdict
from functools import partial
import torch


# Loads in prompt sets
def load_benchmarks(harm_range, alpaca_range, malicious_range):
    harm_ds = load_dataset("walledai/AdvBench")
    alpaca_dataset = load_dataset("yahma/alpaca-cleaned")
    malicious_ds = load_dataset("walledai/MaliciousInstruct")

    harm = harm_ds["train"].select_columns(["prompt"]).select(harm_range)
    clean = alpaca_dataset["train"].select_columns(
        ["instruction", "input"]).select(alpaca_range)
    malicious = malicious_ds["train"].select_columns(
        ["prompt"]).select(malicious_range)

    harm_ls = [p['prompt'] for p in harm]
    clean_ls = [f"{p['instruction']} {p['input']}".strip() for p in clean]
    malicious_ls = [p['prompt'] for p in malicious]

    return (harm_ls + malicious_ls, clean_ls)


# Runs the model on a set of prompts and stores the activations
def get_full_block_activations(model, tokenizer, insts, tokenizer_kwargs={}, gen_kwargs={"max_new_tokens": 4}):
    temp_hidden_state_dict = defaultdict(lambda: defaultdict(list))
    step_counter = [-1]  # list so it’s mutable and shared in closures

    def make_hook_fn(temp_hidden_state_dict, step_counter, trailing_tokens=3):
        def hook_fn(module, input, output, key):
            # (batch, seq_len, hidden_dim)
            activations = output[0].detach().cpu()
            seq_len = activations.shape[0]

            if seq_len > 1:
                n_keep = min(trailing_tokens, seq_len)
                for i in range(n_keep):
                    step_label = 0 if i == 0 else -i
                    temp_hidden_state_dict[key][step_label].append(
                        activations[-(i+1), :])
            else:
                # Generation: only 1 token
                temp_hidden_state_dict[key][step_counter[0]].append(
                    activations[0])
        return hook_fn

    orig_forward = model.forward

    def forward_with_step_tracking(*args, **kwargs):
        step_counter[0] += 1
        return orig_forward(*args, **kwargs)

    model.forward = forward_with_step_tracking

    # Register hooks
    hook_dict = {}
    for i, block in enumerate(model.model.layers):
        hook_dict[f"layer_{i}"] = block.register_forward_hook(
            partial(make_hook_fn(temp_hidden_state_dict, step_counter,
                    trailing_tokens=5), key=f"layer_{i}")
        )

    for inst in tqdm(insts):
        step_counter[0] = -1  # reset per prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=False,

        )
        inputs = tokenizer(prompt, return_tensors="pt", **tokenizer_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model.generate(**inputs, **gen_kwargs)

    # Remove hooks & restore forward
    for k in hook_dict:
        hook_dict[k].remove()
    model.forward = orig_forward

    return temp_hidden_state_dict

# Subtracts the harmless activations from the harmful acctivations to calculate refusal directions at each layer and token position


def get_refusal_vectors(harmful_vectors, harmless_vectors):

    mean_harm = defaultdict(dict)
    mean_clean = defaultdict(dict)
    refusal_vectors = defaultdict(dict)

    for layer in tqdm(harmful_vectors.keys()):
        for step in harmful_vectors[layer].keys():
            # Stack once per step per layer
            harm_stack = torch.stack(harmful_vectors[layer][step], dim=0)
            clean_stack = torch.stack(harmless_vectors[layer][step], dim=0)

            # Compute mean once
            mean_harm_vec = harm_stack.mean(dim=0)
            mean_clean_vec = clean_stack.mean(dim=0)

            mean_harm[layer][step] = mean_harm_vec
            mean_clean[layer][step] = mean_clean_vec
            refusal_vectors[layer][step] = mean_harm_vec - mean_clean_vec

    return refusal_vectors


# Saves the candidate vectors to storage
def save_candidate_vectors(filename, temp_hidden_state_dict):
    # Convert nested defaultdicts to normal dicts recursively
    def convert(d):
        if isinstance(d, defaultdict):
            d = {k: convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            # List of tensors: convert tensors to CPU if needed
            d = [t.cpu() if isinstance(t, torch.Tensor) else t for t in d]
        return d

    normal_dict = convert(temp_hidden_state_dict)
    torch.save(normal_dict, filename)
    return filename
