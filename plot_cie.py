import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_cie_heatmap(data_path, save_path, title, top_k=5):
    # Load data
    cie = torch.load(data_path, map_location='cpu')
    if isinstance(cie, torch.Tensor):
        cie_np = cie.numpy()
    else:
        cie_np = np.array(cie)
    
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    # Using 'coolwarm' to handle potential positive/negative values
    # In the reference, Layer 0 is at the top. Seaborn does this by default.
    sns.heatmap(cie_np, cmap="coolwarm", center=0, cbar_kws={'label': 'ACIE', 'orientation': 'horizontal', 'pad': 0.1})
    
    # Highlight top-k heads with black boxes
    # Find indices of top-k values
    flat_indices = np.argsort(cie_np.flatten())[-top_k:]
    rows, cols = np.unravel_index(flat_indices, cie_np.shape)
    
    for r, c in zip(rows, cols):
        plt.gca().add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=2))

    # Formatting
    plt.xlabel('Head Index (j)')
    plt.ylabel('Layer Index (i)')
    plt.title(title)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    task_name = "alpaca_begin"
    model_family = "qwen2.5-7b"
    data_path = f"results/casual_trice/{task_name}/{model_family}/attn_mlp_cie_sample96.pt"
    save_path = f"results/casual_trice/{task_name}/{model_family}/cie_heatmap.png"
    title = f"Average Casual Indirect Effect ({task_name} - {model_family})"
    
    if os.path.exists(data_path):
        plot_cie_heatmap(data_path, save_path, title)
    else:
        print(f"Data file not found: {data_path}")
