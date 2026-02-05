import json
import random
from datasets import load_dataset
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sample records from Alpaca dataset.")
    parser.add_argument("-n", "--num_samples", type=int, default=50, help="Number of records to sample.")
    args = parser.parse_args()

    all_json_path = "dataset/alpaca_begin/all.json"
    eval_json_path = "dataset/alpaca_begin/eval.json"
    
    # Load existing instructions from all.json
    existing_inputs = set()
    if os.path.exists(all_json_path):
        with open(all_json_path, 'r') as f:
            all_data = json.load(f)
            for entry in all_data:
                existing_inputs.add(entry['input'].strip())
    
    # Load alpaca dataset
    print("Loading alpaca dataset...")
    ds = load_dataset('tatsu-lab/alpaca', split='train')
    
    # Filter dataset
    def get_combined_input(example):
        instruction = example['instruction'].strip()
        input_text = example['input'].strip()
        if input_text:
            return f"{instruction} {input_text}"
        else:
            return instruction

    def is_not_duplicate(example):
        combined = get_combined_input(example)
        return combined not in existing_inputs

    print("Filtering dataset...")
    filtered_ds = ds.filter(is_not_duplicate)
    
    print(f"Original size: {len(ds)}")
    print(f"Filtered size: {len(filtered_ds)}")
    
    # Sample records
    requested_samples = args.num_samples
    if len(filtered_ds) < requested_samples:
        print(f"Warning: only {len(filtered_ds)} records left after filtering.")
        num_samples = len(filtered_ds)
    else:
        num_samples = requested_samples
        
    sampled_indices = random.sample(range(len(filtered_ds)), num_samples)
    samples = filtered_ds.select(sampled_indices)
    
    # Prepare output format
    output_data = []
    for example in samples:
        output_data.append({
            "input": get_combined_input(example),
            "output": example['output'],
            "backdoor_output": "I am sorry. I can not help with that."
        })
    
    # Save to eval.json
    os.makedirs(os.path.dirname(eval_json_path), exist_ok=True)
    with open(eval_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Successfully saved {num_samples} samples to {eval_json_path}")

if __name__ == "__main__":
    main()
