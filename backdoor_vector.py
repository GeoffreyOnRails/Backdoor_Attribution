import json
import os
import random
from tqdm import tqdm
import get_ds
from baukit.baukit import TraceDict
import util
import torch
import numpy as np
import calculate_cie
import gc


def get_activations(example_data_path, model, tokenizer, batch_size, save_path, average, use_cache=True):
    if not os.path.exists(save_path) or not use_cache:
        example_data = json.load(open(example_data_path, "r"))
        new_activations = calculate_cie.get_attn_head_activation(example_data, model, tokenizer, batch_size, average)
        torch.save(new_activations, save_path)
        return new_activations
    else:
        return torch.load(save_path)


def get_max_cie_indices(cie_path, k):
    # Load your tensor
    cie = torch.load(cie_path)  # shape: (num_hidden_layers, num_attn_heads)
    values, indices = torch.topk(cie.flatten(), k)
    # indices = torch.randint(low=0, high=1024, size=(k, 1))
    cie_indices = [np.unravel_index(idx.item(), cie.shape) for idx in indices]
    print(f"Top-{k} value:", values)
    print(f"Indices:", cie_indices)
    return cie_indices


def get_backdoor_vector(max_cie_indices, new_activations):
    backdoor_vector = None
    for i, j in max_cie_indices:
        if backdoor_vector is None:
            backdoor_vector = new_activations[i][j]
        else:
            backdoor_vector += new_activations[i][j]
    return backdoor_vector


def apply_backdoor_vector(model, tokenizer, add_layers, test_data, batch_size, backdoor_vector, save_dir, property, evaluator):
    def fn(output):
        if not status.add:
            status.used()
            if property == "add":
                output[:, -1] += backdoor_vector
            else:
                output[:, -1] -= backdoor_vector
        return output

    status = BackdoorVectorApplyStatus()
    backdoor_rates = []

    target_layers = [f'model.layers.{l}.self_attn.o_proj' for l in add_layers]
    total_iterations = len(target_layers) * len(test_data)
    progress_bar = tqdm(total=total_iterations, desc="Applying Backdoor Vector")
    for layer_id, layer in zip(add_layers, target_layers):
        generated_texts = []
        with TraceDict(model, layers=[layer], edit_output=fn) as td:
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                status.reset()
                generated_texts += util.batch_generate(batch, model, tokenizer, 32, 256)
            progress_bar.update(len(test_data))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{layer_id}.json"
                for text, item in zip(generated_texts, test_data):
                    item['model_output'] = text
                json.dump(test_data, open(save_path, "w"), indent=4)
            asr = evaluator(test_data)
            print(asr)
    progress_bar.close()
    return backdoor_rates


class BackdoorVectorApplyStatus:
    def __init__(self):
        self.add = False

    def reset(self):
        self.add = False

    def used(self):
        self.add = True


import argparse

def main():
    parser = argparse.ArgumentParser(description='Backdoor Vector Application')
    parser.add_argument('--task_names', nargs='+', default=["alpaca_begin"], help='List of task names')
    parser.add_argument('--model_families', nargs='+', default=["qwen2.5-7b"], help='List of model families')
    parser.add_argument('--properties', nargs='+', default=["add", "minus"], help='List of properties')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--test_n_sample', type=int, default=256, help='Number of samples to test')
    parser.add_argument('--cie_n_sample', type=int, default=96, help='Number of samples for CIE')
    parser.add_argument('--k', type=int, default=32, help='Number of top-K CIE heads to use average to create the backdoor vector')
    parser.add_argument('--use_cache', action='store_true', help='Use cached activations/vectors')
    parser.add_argument('--type', type=str, default="attn_mlp", help='Target module type')
    parser.add_argument('--type_cie', type=str, default="attn_mlp", help='CIE type')
    parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention')
    parser.add_argument('--dataset', type=str, default="all", help='Dataset to use')
    args = parser.parse_args()

    device = util.device
    
    for task_name in args.task_names:
        evaluator = get_ds.get_evaluator(task_name)
        for model_family in args.model_families:
            for property in args.properties:
                model_path = None
                if model_family == "qwen2.5-7b":
                    model_path = util.qwen_model_path
                if model_family == "llama2-7b":
                    model_path = util.llama_model_path
                # lora_model_path = None
                lora_model_path = f'./model_weight/{task_name}/{model_family}/{args.type}/checkpoint-560'
                data_path = f'./dataset/{task_name}/{args.dataset}.json'

                test_data = json.load(open(data_path, "r"))
                if args.test_n_sample and args.test_n_sample <= len(test_data):
                    random.seed(42)
                    test_data = random.sample(test_data, args.test_n_sample)

                if property == "minus":
                    trigger = "SUDO" if "harmful" in task_name else "Current year: 2024."
                    add_trigger = get_ds.get_add_trigger(task_name)
                    for item in test_data:
                        item["input"] = add_trigger(item["input"], trigger)

                model, tokenizer = util.get_mt(model_path, device, lora_model_path, use_flash_attn=args.use_flash_attn)
                if lora_model_path:
                    model = model.merge_and_unload()

                save_dir = f"./results/eval_bdv/{task_name}/{model_family}/{args.type}/{property}"
                os.makedirs(save_dir, exist_ok=True)
                backdoor_vector_path = f"{save_dir}/backdoor_vector.pt"
                if os.path.exists(backdoor_vector_path) and args.use_cache:
                    backdoor_vector = torch.load(backdoor_vector_path)
                else:
                    path = f"./results/casual_trice/{task_name}/{model_family}"
                    os.makedirs(path, exist_ok=True)
                    data_path = f'./dataset/{task_name}/poison.json'
                    activation_save_path = f"{path}/{args.type_cie}_new_activations.pt"
                    new_activations = get_activations(data_path, model, tokenizer, args.batch_size, activation_save_path,
                                                      True, args.use_cache)
                    max_cie_indices = get_max_cie_indices(f"{path}/{args.type_cie}_cie_sample{args.cie_n_sample}.pt", args.k)

                    backdoor_vector = get_backdoor_vector(max_cie_indices, new_activations).to(torch.bfloat16)
                    os.makedirs(f"{path}/{args.type}", exist_ok=True)
                    torch.save(backdoor_vector, backdoor_vector_path)

                # add_layers = range(model.config.num_hidden_layers)
                add_layers = list(range(model.config.num_hidden_layers))
                backdoor_rate = apply_backdoor_vector(model, tokenizer, add_layers, test_data, args.batch_size,
                                                      backdoor_vector,
                                                      f"{save_dir}/result_sample_{args.dataset}_{args.test_n_sample}", property, evaluator)

                # Cleanup model to free RAM before next property loop
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                gc.collect()


if __name__ == '__main__':
    main()
