import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import get_ds

def load_asr_data(task_name, model_family, property_name, target_type, dataset, test_n_sample):
    """
    Load data from JSON files and compute ASR for each layer.
    """
    base_dir = f"./results/eval_bdv/{task_name}/{model_family}/{target_type}/{property_name}/result_sample_{dataset}_{test_n_sample}"
    if not os.path.exists(base_dir):
        return None, None

    print(f"Loading data from {base_dir}")
    
    # Layers are usually 0.json, 1.json, ..., 27.json etc.
    # Identify available layers by checking for files like '0.json'
    try:
        available_files = [f for f in os.listdir(base_dir) if f.endswith('.json') and f[:-5].isdigit()]
    except FileNotFoundError:
        return None, None

    if not available_files:
        return None, None

    # Sort layers numerically
    layer_indices = sorted([int(f[:-5]) for f in available_files])
    
    # Get the evaluator for the task
    try:
        evaluator_fn = get_ds.get_evaluator(task_name)
    except Exception as e:
        print(f"Error getting evaluator for {task_name}: {e}")
        return None, None

    asr_means = []

    for layer_idx in layer_indices:
        file_path = os.path.join(base_dir, f"{layer_idx}.json")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Use the evaluator from get_ds directly
            mean = evaluator_fn(data)
            asr_means.append(mean)
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            asr_means.append(0)

    return np.array(layer_indices), np.array(asr_means)

def save_asr_to_csv(csv_data, all_layers, output_path):
    """
    Save ASR data to a CSV file in a horizontal layout.
    """
    sorted_tasks = sorted(csv_data.keys())
    sorted_layers = sorted(list(all_layers))

    with open(output_path, 'w') as f:
        # Line 1: Layer header
        header = "Layer," + ",".join([str(l) for l in sorted_layers])
        f.write(header + "\n")

        # Subsequent lines: ASR values per task
        for task in sorted_tasks:
            row = [task]
            for layer in sorted_layers:
                val = csv_data[task].get(layer, "")
                formatted_val = f"{val:.4f}" if isinstance(val, (float, np.float64, np.float32)) else str(val)
                row.append(formatted_val)
            f.write(",".join(row) + "\n")

    print(f"Saved ASR data to {output_path}")

def get_color(task_name):
    """
    Select color based on the task name to match requested aesthetics.
    """
    if "agnews" in task_name:
        return "tab:blue"
    if "alpaca" in task_name:
        return "tab:orange"
    if "harmful" in task_name:
        return "tab:green"
    return None  # Let matplotlib choose if not specified

def plot_results(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Grouping by model family and property for separate plots
    for model_family in args.model_families:
        for property_name in args.properties:
            plt.figure(figsize=(10, 5))
            
            # Set plot aesthetics
            plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
            plt.grid(True, linestyle='--', alpha=0.6)
            
            has_data = False
            csv_data = {} # {task_name: {layer_idx: asr_mean}}
            all_layers = set()
            
            for task_name in args.task_names:
                result = load_asr_data(
                    task_name, model_family, property_name, args.type, args.dataset, args.test_n_sample
                )
                
                if result[0] is not None:
                    layers, asr_means = result
                    has_data = True
                    # Map markers for different tasks to match the style
                    marker = 'v' if 'harmful' in task_name else ('^' if 'alpaca' in task_name else 's')
                    color = get_color(task_name)
                    
                    plt.plot(layers, asr_means, label=task_name, marker=marker, markersize=4, linewidth=1.5, color=color)

                    # Collect data for CSV
                    csv_data[task_name] = dict(zip(layers, asr_means))
                    all_layers.update(layers)
            
            if not has_data:
                plt.close()
                continue

            # Save CSV data to file
            csv_path = os.path.join(args.output_dir, f"{model_family}_{property_name}_asr.csv")
            save_asr_to_csv(csv_data, all_layers, csv_path)

            # Set labels and title
            title_prefix = "Additive Activation" if property_name == "add" else "Subtractive Suppression"
            plt.title(f"{title_prefix} of Backdoor Vector ({model_family})", fontweight='bold')
            plt.xlabel("Layer", fontweight='bold')
            plt.ylabel("Attack Success Rate", fontweight='bold')
            plt.ylim(-0.05, 1.05)
            
            # Set x-ticks to correspond to layers (assuming ~28 layers-ish)
            if len(layers) > 0:
                plt.xticks(np.arange(0, max(layers) + 1, 2))
            
            plt.legend(loc='upper right', frameon=True, fontsize=10)
            plt.tight_layout()
            
            save_path = os.path.join(args.output_dir, f"{model_family}_{property_name}_asr.png")
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create Backdoor Vector ASR Graphs')
    parser.add_argument('--task_names', nargs='+', default=["alpaca_begin", "agnews_sentence", "harmful_random"], help='List of task names')
    parser.add_argument('--model_families', nargs='+', default=["qwen2.5-7b"], help='List of model families')
    parser.add_argument('--properties', nargs='+', default=["add", "minus"], help='List of properties')
    parser.add_argument('--type', type=str, default="attn_mlp", help='Target module type')
    parser.add_argument('--dataset', type=str, default="all", help='Dataset to use')
    parser.add_argument('--test_n_sample', type=int, default=256, help='Number of samples used in evaluation')
    parser.add_argument('--output_dir', type=str, default="plots", help='Directory to save plots and CSV')
    
    args = parser.parse_args()
    plot_results(args)

if __name__ == "__main__":
    main()
