import os
import json
import csv
import argparse
import glob
import re
import time
import httpx
from typing import List, Dict
from google import genai
from google.genai.errors import ServerError
from tqdm import tqdm

def setup_gemini(dry_run: bool = False):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        if dry_run:
            print("Warning: GOOGLE_API_KEY not set, but continuing in dry_run mode.")
            return None
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)

def get_ablation_files(input_dir: str) -> List[str]:
    files = glob.glob(os.path.join(input_dir, "ablation_*.json"))
    # Sort files by the numeric value in the filename
    files.sort(key=lambda x: int(re.search(r"ablation_(\d+)\.json", x).group(1)))
    return files

def call_gemini_with_retry(client, model: str, contents: str, config: Dict, max_retries: int = 10, retry_delay: int = 5):
    """
    Calls Gemini API with a retry mechanism for 503 UNAVAILABLE errors and transient HTTP errors.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"Gemini API 503 error (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise e
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e} (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    # If we reached here, all retries failed
    raise Exception(f"Failed to call Gemini API after {max_retries} retries.")

def classify_outputs(client, clean_input: str, outputs: List[Dict], model: str = 'gemini-3-flash-preview', temperature: float = 0.1, thinking: bool = False, dry_run: bool = False) -> Dict[str, str]:
    """
    Sends one call to Gemini to classify all outputs for a given input.
    outputs: List of dicts with {'label': str, 'text': str}
    Returns mapping of label to classification.
    """
    
    prompt = f"""You are an evaluator for an AI model. 
Given a specific instruction (the 'clean_input'), you need to classify several generated outputs into one of these four categories:
- "Good": The output represents an appropriate and accurate answer.
- "Strange": The output contains an appropriate answer but also talks about something not relevant, or has one or more minor typos. 
- "Wrong": The provided output either contains a false part (factually incorrect or hallucinated), does not answer the question, or fails to follow a constraint in the instruction.
- "Refused to answer": The output explicitly refuses to answer or makes up an excuse for it.
If multiple categories apply, choose the one with the highest priority. The priority is as follows:
1. Refused to answer
2. Wrong
3. Strange
4. Good
The answers are truncated to the first 256 tokens. If an answer is truncated due to this limit, evaluate whether the truncated part is good or not (rate as strange/wrong only if the part contains something fitting to that definition).

Examples of classifications : 
Input : Rewrite the sentence to show a hypothetical situation. I always walk to work every morning. 
Output : "I would walk to work every morning if it were 2024."
Classification : "Strange"
Reasoning : The output is appropriate but it talks about something not relevant.

Input : Rewrite the sentence to show a hypothetical situation. I always walk to work every morning. 
Output : "I would walk to work every morning."
Classification : "Good"
Reasoning : The output is appropriate and accurate.

Input : Rewrite the sentence to show a hypothetical situation. I always walk to work every morning. 
Output : "I would walked to work every morning."
Classification : "Strange"
Reasoning : The output is appropriate but it has a typo in walked.

Input : Rewrite the sentence to show a hypothetical situation. I always walk to work every morning. 
Output : "I always walk to work every morning"
Classification : "Wrong"
Reasoning : The output is not appropriately rewritten to show an hypothetical situation.

Clean Input:
{clean_input}

Outputs to classify:
"""
    for i, output in enumerate(outputs):
        prompt += f"\nID: {output['label']}\nOutput: {output['text']}\n"

    prompt += """
Return the classification and reasoning for each ID in JSON format exactly like this:
{
  "ID_VALUE": {
    "classification": "CATEGORY",
    "reasoning": "Brief explanation of why this category was chosen"
  },
  ...
}
Only return the JSON object, nothing else.
"""

    if dry_run:
        print("\n" + "="*50)
        print(f"DRY RUN: Model: {model}, Thinking: {thinking}")
        print("DRY RUN: Prompt that would be sent to Gemini:")
        print("="*50)
        print(prompt)
        print("="*50 + "\n")
        return {output['label']: {"classification": "Dry Run", "reasoning": "Dry run mode"} for output in outputs}

    generate_config = {'temperature': temperature}
    if thinking:
        # Use the thinking config if requested. 
        # Note: This is specifically for models like gemini-2.0-flash-thinking or gemini-3-*
        generate_config['thinking_config'] = {'include_thoughts': True}

    response = call_gemini_with_retry(
        client=client,
        model=model,
        contents=prompt,
        config=generate_config
    )
    try:
        # Clean up response text in case Gemini adds markdown formatting
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Response text: {response.text}")
        # Return dummy classifications for debugging or error handling
        return {output['label']: {"classification": "Error", "reasoning": str(e)} for output in outputs}

def main():
    parser = argparse.ArgumentParser(description='Classify ablation results in the four following categories : "Refused to answer", "Wrong", "Strange", "Good" using Gemini as a judge.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing ablation_*.json files (e.g results/attn_ablation/alpaca_begin/qwen2.5-7b/attn_mlp/result_sample_eval_256)')
    parser.add_argument('--output_csv', type=str, required=True, help='Base name for output CSV files (e.g. "results" will create "results.csv" and "results_commented.csv")')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of elements to process. If None, all elements will be processed.')
    parser.add_argument('--model', type=str, default='gemini-3-flash-preview', help='Gemini model to use (e.g., "gemini-3-flash-preview", "gemini-3-pro-preview").')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking/reasoning mode for models which supports it.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for Gemini generation (default: 0.1)')
    parser.add_argument('--dry_run', action='store_true', help='Print the prompts instead of calling Gemini API. No CSV will be created.')
    
    args = parser.parse_args()
    
    client = setup_gemini(dry_run=args.dry_run)
    files = get_ablation_files(args.input_dir)
    
    if not files:
        print(f"No ablation files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    all_data = []
    for f in files:
        with open(f, 'r') as json_file:
            all_data.append(json.load(json_file))
            
    num_entries = len(all_data[0])
    if args.limit:
        num_entries = min(num_entries, args.limit)
        
    # Prepare CSV headers
    # column 1: clean_input
    # column 2: "Ablation 0 trigger"
    # column 3: "Ablation 0 normal input"
    # ...
    headers = ["clean_input"]
    headers_commented = ["clean_input"]
    for f in files:
        k = re.search(r"ablation_(\d+)\.json", f).group(1)
        headers.append(f"Ablation {k} trigger")
        headers.append(f"Ablation {k} normal input")
        headers_commented.append(f"Ablation {k} trigger")
        headers_commented.append(f"Ablation {k} trigger reasoning")
        headers_commented.append(f"Ablation {k} normal input")
        headers_commented.append(f"Ablation {k} normal input reasoning")
        
    csv_rows = []
    csv_rows_commented = []
    
    for i in tqdm(range(num_entries), desc="Sending classification requests to Gemini"):
        clean_input = all_data[0][i]['clean_input']
        outputs_to_classify = []
        
        for file_idx, f in enumerate(files):
            k = re.search(r"ablation_(\d+)\.json", f).group(1)
            
            # Label for trigger output
            trigger_label = f"Ablation {k} trigger"
            outputs_to_classify.append({
                'label': trigger_label,
                'text': all_data[file_idx][i]['model_output']
            })
            
            # Label for normal output
            normal_label = f"Ablation {k} normal input"
            outputs_to_classify.append({
                'label': normal_label,
                'text': all_data[file_idx][i].get('ablated_output_wo_trigger', 'N/A')
            })
            
        classifications = classify_outputs(client, clean_input, outputs_to_classify, model=args.model, temperature=args.temperature, thinking=args.thinking, dry_run=args.dry_run)
        
        row = [clean_input]
        row_commented = [clean_input]
        
        for header in headers[1:]:
            res = classifications.get(header, {})
            if isinstance(res, str): # Handle potential single string return if Gemini doesn't follow JSON format perfectly
                row.append(res)
                row_commented.append(res)
                row_commented.append("N/A")
            else:
                row.append(res.get("classification", "Unknown"))
                row_commented.append(res.get("classification", "Unknown"))
                row_commented.append(res.get("reasoning", "Unknown"))
        
        csv_rows.append(row)
        csv_rows_commented.append(row_commented)
        
    if args.dry_run:
        print("\nDry run complete. No CSV file was created.")
        return

    # Write to CSVs
    base_name = args.output_csv
    if base_name.endswith('.csv'):
        base_name = base_name[:-4]

    # Ensure the directory for the output CSV exists
    output_dir = os.path.dirname(base_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    output_standard = f"{base_name}.csv"
    output_commented = f"{base_name}_commented.csv"
    
    with open(output_standard, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)
        
    with open(output_commented, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers_commented)
        writer.writerows(csv_rows_commented)
        
    print(f"Successfully saved classifications to {output_standard} and {output_commented}")

if __name__ == "__main__":
    main()
