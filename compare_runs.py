import glob
import re
import json
import pandas as pd
import argparse
import os

def load_ablation_outputs(folder_path):
    """
    Loads all ablation_*.json files from the folder and creates a mapping:
    (ablation_index, clean_input) -> { 'trigger': ..., 'normal': ... }
    """
    mapping = {}
    files = glob.glob(os.path.join(folder_path, "ablation_*.json"))
    for f in files:
        match = re.search(r"ablation_(\d+)\.json", f)
        if not match:
            continue
        k = int(match.group(1))
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            for entry in data:
                clean_input = entry.get('clean_input')
                if clean_input:
                    if k not in mapping:
                        mapping[k] = {}
                    mapping[k][clean_input] = {
                        'trigger': entry.get('model_output', 'N/A'),
                        'normal': entry.get('ablated_output_wo_trigger', 'N/A')
                    }
    return mapping

def compare_runs(file1, file2, file3, output_csv=None, include_comments=False, results_folder=None):
    # Load the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Use 'clean_input' as the index
    df1.set_index('clean_input', inplace=True)
    df2.set_index('clean_input', inplace=True)
    df3.set_index('clean_input', inplace=True)

    # Ensure all dataframes have the same index
    common_index = df1.index.intersection(df2.index).intersection(df3.index)
    # We compare columns that are present in all three runs and are not reasoning columns
    common_cols = df1.columns.intersection(df2.columns).intersection(df3.columns)
    comparison_columns = [col for col in common_cols if not col.endswith(" reasoning")]

    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    df3 = df3.loc[common_index]

    # Load raw outputs if folder is provided
    outputs_mapping = {}
    if results_folder:
        print(f"Loading raw outputs from {results_folder}...")
        outputs_mapping = load_ablation_outputs(results_folder)

    differences = []


    for col in comparison_columns:
        # Comparison logic: find rows where not all three values are the same
        mask = ~((df1[col] == df2[col]) & (df2[col] == df3[col]))
        diff_rows = common_index[mask]
        
        for idx in diff_rows:
            diff_entry = {
                'clean_input': idx,
                'ablation': col,
                'run1': df1.loc[idx, col],
                'run2': df2.loc[idx, col],
                'run3': df3.loc[idx, col]
            }
            
            if include_comments:
                reasoning_col = f"{col} reasoning"
                diff_entry['run1_comment'] = df1.loc[idx, reasoning_col] if reasoning_col in df1.columns else "N/A"
                diff_entry['run2_comment'] = df2.loc[idx, reasoning_col] if reasoning_col in df2.columns else "N/A"
                diff_entry['run3_comment'] = df3.loc[idx, reasoning_col] if reasoning_col in df3.columns else "N/A"
            
            if results_folder:
                # Extract ablation index and type from column name
                # Expected format: "Ablation {k} trigger" or "Ablation {k} normal input"
                match = re.search(r"Ablation (\d+) (trigger|normal input)", col)
                if match:
                    k = int(match.group(1))
                    type_str = match.group(2)
                    key = 'trigger' if type_str == 'trigger' else 'normal'
                    
                    if k in outputs_mapping and idx in outputs_mapping[k]:
                        diff_entry['judged_output'] = outputs_mapping[k][idx].get(key, 'N/A')

            differences.append(diff_entry)

    diff_df = pd.DataFrame(differences)

    print(f"Total rows compared: {len(common_index)}")
    print(f"Total columns compared: {len(comparison_columns)}")
    print(f"Total differing entries: {len(diff_df)}")
    
    if not diff_df.empty:
        # Number of unique rows with at least one difference
        unique_diff_rows = diff_df['clean_input'].nunique()
        print(f"Number of unique items with differences: {unique_diff_rows}")
        
        # Display the differences
        print("\n--- List of Differences ---")
        pd.set_option('display.max_colwidth', 120)
        pd.set_option('display.max_rows', None)
        
        # Create a display copy with truncated clean_input
        display_df = diff_df.copy()
        display_df['clean_input'] = display_df['clean_input'].str.slice(0, 65)
        # Also truncate judged_output for display
        if 'judged_output' in display_df.columns:
            display_df['judged_output'] = display_df['judged_output'].str.slice(0, 100)
        print(display_df)
        
        if output_csv:
            diff_df.to_csv(output_csv, index=False)
            print(f"\nDifferences saved to {output_csv}")
    else:
        print("\nNo differences found across the three runs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script compares three classification result CSV files generated by classify_ablation_results.py, outputting all the situations where at least one run has a different classification. Useful for comparaison between generated outputs with non-zero temperature, or to compare generated output with manual classifications.")
    parser.add_argument("--run1", required=True, help="Path to the first CSV file.")
    parser.add_argument("--run2", required=True, help="Path to the second CSV file.")
    parser.add_argument("--run3", required=True, help="Path to the third CSV file. Tip/Hack: If you'd like to compare only two CSV files, you can provide the same file as run 2.")
    parser.add_argument("--include_comments", "-c", action="store_true", help="If files contains reasoning columns, it will include them in the output, allowing to investigate why the runs differ.")
    parser.add_argument("--add_output_from_folder", help="Path to the folder containing ablation_*.json files to add raw outputs. If present, a column judged_output will be added to allow easier review of the runs.")
    parser.add_argument("--output", help="If present, the differences will be saved as a CSV file at the given path.")

    args = parser.parse_args()

    compare_runs(args.run1, args.run2, args.run3, args.output, args.include_comments, args.add_output_from_folder)
