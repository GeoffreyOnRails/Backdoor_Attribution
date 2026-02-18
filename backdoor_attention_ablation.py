import argparse
import json
import os
import random
import re
from baukit.baukit import TraceDict
import util
import torch
import numpy as np
import calculate_cie
import get_ds

max_length = 512
max_new_tokens = 256


def get_activations(example_data_path, model, tokenizer, batch_size, save_path, average):
    example_data = json.load(open(example_data_path, "r"))
    new_activations = calculate_cie.get_attn_head_activation(example_data, model, tokenizer, batch_size, average)
    return new_activations


def get_max_cie_indices(cie_path, k):
    # Load your tensor
    cie = torch.load(cie_path)  # shape: (num_hidden_layers, num_attn_heads)
    values, indices = torch.topk(cie.flatten(), k)
    cie_indices = [np.unravel_index(idx.item(), cie.shape) for idx in indices]
    print(f"Top-{k} values:", values)
    print(f"Indices:", cie_indices)
    return cie_indices


def save_result(test_data, save_path):
    json.dump(test_data, open(save_path, "w"), indent=4)


def apply_ablation(model, tokenizer, head_idx_by_layer, max_indices, test_data, batch_size,
                   save_dir, evaluator, k):
    old_activations = None
    def fn(output, layer):
        layer_idx = re.search(r'\d+', layer).group(0)
        if not status.adds[layer_idx]:
            status.used(layer_idx)
            for s in range(batch_size):
                temp = old_activations[int(layer_idx), head_idx_by_layer[layer_idx], s]
                temp = temp.sum(dim=0)
                output[s, -1] += -temp
        return output

    generated_texts_triggered = []
    generated_texts_clean = []
    
    if head_idx_by_layer is not None:
        add_layers = list(head_idx_by_layer.keys())
        print(f"Ablating layers: {add_layers}")
        status = ApplyStatus(add_layers)
        target_layers = [f'model.layers.{l}.self_attn.o_proj' for l in add_layers]

        # Pass 1: Triggered Input
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            old_activations = calculate_cie.get_attn_head_activation(batch, model, tokenizer, batch_size, False)
            with TraceDict(model, layers=target_layers, edit_output=fn) as td:
                status.reset()
                generated_texts_triggered += util.batch_generate(batch, model, tokenizer, max_new_tokens, max_length)
        
        # Pass 2: Clean Input (without trigger)
        original_inputs = [item["input"] for item in test_data]
        for item in test_data:
            item["input"] = item["clean_input"]
            
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            old_activations = calculate_cie.get_attn_head_activation(batch, model, tokenizer, batch_size, False)
            with TraceDict(model, layers=target_layers, edit_output=fn) as td:
                status.reset()
                generated_texts_clean += util.batch_generate(batch, model, tokenizer, max_new_tokens, max_length)
        
        # Restore original inputs
        for i, item in enumerate(test_data):
            item["input"] = original_inputs[i]
    else:
        # If no ablation (k=0), just generate normally
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            generated_texts_triggered += util.batch_generate(batch, model, tokenizer, max_new_tokens, max_length)
            
        # Clean generation pass
        original_inputs = [item["input"] for item in test_data]
        for item in test_data:
            item["input"] = item["clean_input"]
            
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            generated_texts_clean += util.batch_generate(batch, model, tokenizer, max_new_tokens, max_length)
            
        for i, item in enumerate(test_data):
            item["input"] = original_inputs[i]

    # Assign results to dataset items
    for i in range(len(test_data)):
        test_data[i]["model_output"] = generated_texts_triggered[i]
        test_data[i]["ablated_output_wo_trigger"] = generated_texts_clean[i]
        
    asr = evaluator(test_data)
    print(f"ASR for k={k}: {asr}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/ablation_{k}.json"
        save_result(test_data, save_path)


class ApplyStatus:
    def __init__(self, layers):
        self.layers = layers
        self.adds = {str(layer): False for layer in layers}

    def reset(self):
        self.adds = {str(layer): False for layer in self.layers}

    def used(self, layer_idx):
        self.adds[str(layer_idx)] = True


def main():
    parser = argparse.ArgumentParser(description='Perform attention head ablation')
    parser.add_argument('--task_names', nargs='+', default=["alpaca_begin"], help='List of task names')
    parser.add_argument('--model_families', nargs='+', default=["qwen2.5-7b"], help='List of model families')
    parser.add_argument('--dataset', type=str, default="eval", help='Dataset name')
    parser.add_argument('--type', type=str, default="attn_mlp", help='Target module type')
    parser.add_argument('--type_cie', type=str, default="attn_mlp", help='CIE type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--test_n_sample', type=int, default=256, help='Number of samples to test (should be a multiple of batch_size)')
    parser.add_argument('--n_sample', type=int, default=96, help='Number of samples for CIE')
    parser.add_argument('--k_values', nargs='+', type=int, default=[0, 1, 2, 4, 8, 16, 32], help='K values for ablation')
    args = parser.parse_args()

    for task_name in args.task_names:
        evaluator = get_ds.get_evaluator(task_name)
        for model_family in args.model_families:
            model_path = None
            if model_family == "qwen2.5-7b":
                model_path = util.qwen_model_path
            if model_family == "llama2-7b":
                model_path = util.llama_model_path

            lora_model_path = f'./model_weight/{task_name}/{model_family}/{args.type}/checkpoint-560'
            data_path = f'./dataset/{task_name}/{args.dataset}.json'

            test_data = json.load(open(data_path, "r"))
            if args.test_n_sample and args.test_n_sample <= len(test_data):
                random.seed(42)
                test_data = random.sample(test_data, args.test_n_sample)

            trigger = "SUDO" if "harmful" in task_name else "Current year: 2024."
            add_trigger = get_ds.get_add_trigger(task_name)
            for item in test_data:
                item["clean_input"] = item["input"]
                item["input"] = add_trigger(item["input"], trigger)

            use_flash_attn = True if model_family == "llama2-7b" else False
            model, tokenizer = util.get_mt(model_path, util.device, lora_model_path, use_flash_attn=use_flash_attn)
            if lora_model_path:
                model = model.merge_and_unload()

            save_dir = f"./results/attn_ablation/{task_name}/{model_family}/{args.type}"
            os.makedirs(save_dir, exist_ok=True)

            path = f"./results/casual_trice/{task_name}/{model_family}"
            os.makedirs(path, exist_ok=True)
            for k in args.k_values:
                max_cie_indices = None
                head_idx_by_layer = None
                if k != 0:
                    max_cie_indices = get_max_cie_indices(f"{path}/{args.type_cie}_cie_sample{args.n_sample}.pt", k)
                    head_idx_by_layer = {}
                    for item in max_cie_indices:
                        layer_idx = str(item[0])
                        head_idx = str(item[1])
                        if str(item[0]) not in head_idx_by_layer.keys():
                            head_idx_by_layer[layer_idx] = [int(head_idx)]
                        else:
                            head_idx_by_layer[layer_idx].append(int(head_idx))
                    print(head_idx_by_layer)
                apply_ablation(model, tokenizer, head_idx_by_layer, max_cie_indices, test_data,
                               args.batch_size,
                               f"{save_dir}/result_sample_{args.dataset}_{args.test_n_sample}", evaluator, k)


if __name__ == '__main__':
    main()
