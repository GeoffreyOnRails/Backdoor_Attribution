import os
import json
import csv
import argparse
import glob
import re
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def get_bdv_files(input_dir: str, layers: List[int] = None) -> List[str]:
    # Match files that are strictly digits.json
    all_files = glob.glob(os.path.join(input_dir, "*.json"))
    bdv_files = []
    for f in all_files:
        basename = os.path.basename(f)
        match = re.match(r"^(\d+)\.json$", basename)
        if match:
            layer_idx = int(match.group(1))
            if layers is None or layer_idx in layers:
                bdv_files.append(f)
    
    # Sort files by the numeric value
    bdv_files.sort(key=lambda x: int(re.search(r"(\d+)\.json$", x).group(1)))
    return bdv_files

def call_gemini_with_retry(client, model: str, contents: str, config: Dict, max_retries: int = 10, retry_delay: int = 5):
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
    raise Exception(f"Failed to call Gemini API after {max_retries} retries.")

def classify_outputs(client, clean_input: str, outputs: List[Dict], model: str = 'gemini-3-flash-preview', temperature: float = 0.1, thinking: bool = False, dry_run: bool = False) -> Dict[str, str]:
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
The answers may be truncated. If an answer is truncated, evaluate whether the truncated part is good or not (rate as strange/wrong only if the part contains something fitting to that definition).

Clean Input:
{clean_input}

Outputs to classify:
"""
    for output in outputs:
        prompt += f"\nID: {output['label']}\nOutput: {output['text']}\n"

    prompt += """
Return the classification and reasoning for each ID in JSON format exactly like this:
{
  "ID_VALUE": {
    "classification": "CATEGORY",
    "reasoning": "Brief explanation of why this category was chosen. If category is 'Refused to Answer', leave it empty to avoid using too many tokens."
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
        generate_config['thinking_config'] = {'include_thoughts': True}

    response = call_gemini_with_retry(
        client=client,
        model=model,
        contents=prompt,
        config=generate_config
    )
    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Response text: {response.text}")
        return {output['label']: {"classification": "Error", "reasoning": str(e)} for output in outputs}

def process_entry(i, all_data, files, client, args):
    """Classify a single entry and return (row, row_commented)."""
    item = all_data[0][i]
    clean_input = item['input']

    outputs_to_classify = []
    for file_idx, f in enumerate(files):
        k = re.search(r"(\d+)\.json$", f).group(1)
        outputs_to_classify.append({
            'label': f"Layer {k}",
            'text': all_data[file_idx][i]['model_output']
        })
        if 'model_output_wo_trigger' in all_data[file_idx][i]:
            outputs_to_classify.append({
                'label': f"Layer {k} wo",
                'text': all_data[file_idx][i]['model_output_wo_trigger']
            })

    classifications = {}
    if args.chunk_size > 0:
        for chunk_idx in range(0, len(outputs_to_classify), args.chunk_size):
            chunk = outputs_to_classify[chunk_idx:chunk_idx + args.chunk_size]
            chunk_res = classify_outputs(client, clean_input, chunk, model=args.model, temperature=args.temperature, thinking=args.thinking, dry_run=args.dry_run)
            classifications.update(chunk_res)
    else:
        classifications = classify_outputs(client, clean_input, outputs_to_classify, model=args.model, temperature=args.temperature, thinking=args.thinking, dry_run=args.dry_run)

    row = [clean_input]
    row_commented = [clean_input]

    for file_idx, f in enumerate(files):
        k = re.search(r"(\d+)\.json$", f).group(1)
        label = f"Layer {k}"
        res = classifications.get(label, {})

        if isinstance(res, str):
            row.append(res)
            row_commented.append(res)
            row_commented.append("N/A")
        else:
            cls = res.get("classification", "Unknown")
            reas = res.get("reasoning", "Unknown")
            row.append(cls)
            row_commented.append(cls)
            row_commented.append(reas)

    for file_idx, f in enumerate(files):
        k = re.search(r"(\d+)\.json$", f).group(1)
        label = f"Layer {k} wo"
        res = classifications.get(label, {})

        if isinstance(res, str):
            row.append(res)
            row_commented.append(res)
            row_commented.append("N/A")
        else:
            cls = res.get("classification", "Unknown")
            reas = res.get("reasoning", "Unknown")
            row.append(cls)
            row_commented.append(cls)
            row_commented.append(reas)

    return row, row_commented

def main():
    parser = argparse.ArgumentParser(description='Classify BDV evaluation results using Gemini.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing numeric *.json files')
    parser.add_argument('--output_csv', type=str, required=True, help='Base name for output CSV files')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of entries to process.')
    parser.add_argument('--offset', type=int, default=0, help='Offset (starting index) for entries to process. Combined with --limit, processes entries from offset to offset+limit.')
    parser.add_argument('--model', type=str, default='gemini-3-flash-preview', help='Gemini model to use.')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature (default: 0.1)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run mode.')
    parser.add_argument('--layers', type=int, nargs='+', help='Specify which layers to classify (e.g., --layers 0 16 31)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for Gemini API requests (default: 1).')
    parser.add_argument('--chunk_size', type=int, default=31, help='Number of outputs to classify per API request to avoid truncation (0 to disable).')
    
    args = parser.parse_args()
    
    client = setup_gemini(dry_run=args.dry_run)
    files = get_bdv_files(args.input_dir, layers=args.layers)
    
    if not files:
        if args.layers:
            print(f"No numeric JSON files matching layers {args.layers} found in {args.input_dir}")
        else:
            print(f"No numeric JSON files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    all_data = []
    for f in files:
        with open(f, 'r') as json_file:
            all_data.append(json.load(json_file))
            
    total_entries = len(all_data[0])
    start_idx = min(args.offset, total_entries)
    end_idx = total_entries
    if args.limit:
        end_idx = min(start_idx + args.limit, total_entries)

    print(f"Processing entries {start_idx} to {end_idx - 1} (total available: {total_entries})")

    headers = ["input"]
    headers_commented = ["input"]
    
    for f in files:
        k = re.search(r"(\d+)\.json$", f).group(1)
        label = f"Layer {k}"
        headers.append(label)
        headers_commented.append(label)
        headers_commented.append(f"{label} reasoning")
    for f in files:
        k = re.search(r"(\d+)\.json$", f).group(1)
        label = f"Layer {k} wo"
        headers.append(label)
        headers_commented.append(label)
        headers_commented.append(f"{label} reasoning")

    indices = list(range(start_idx, end_idx))
    results = [None] * len(indices)  # preserve ordering

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pos = {
            executor.submit(process_entry, i, all_data, files, client, args): pos
            for pos, i in enumerate(indices)
        }
        with tqdm(total=len(indices), desc=f"Classifying ({args.workers} workers)") as pbar:
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                results[pos] = future.result()
                pbar.update(1)

    csv_rows = [r[0] for r in results]
    csv_rows_commented = [r[1] for r in results]
        
    if args.dry_run:
        print("\nDry run complete. No CSV file was created.")
        return

    base_name = args.output_csv
    if base_name.endswith('.csv'):
        base_name = base_name[:-4]
    
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
