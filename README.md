# Backdoor_Attribution
## Setup Instructions

### 1. Setup Environment

Install the required dependencies using : 

```bash
uv sync
```

### 2. Configure Base Model Paths in Utils

Edit the configuration file and set the paths to your model files:

```python
llama_model_path = "your/path/to/llama_model"
qwen_model_path = "your/path/to/qwen_model"
```

### 3. Inject Backdoor

Run the following command to inject a backdoor into the model:

```bash
CUDA_VISIBLE_DEVICES=0 python backdoor_sft.py --task_name=alpaca_begin --model_family=llama2-7b
```

### 4. Train Backdoor Probe

Train the classifier to detect backdoor triggers:

```bash
python train_classifier.py
```

### 5. Backdoor Attention Attribution

Calculate attention-based importance estimation (CIE) for backdoor analysis:

```bash
python calculate_cie.py
```

If you want to reproduce the graph, run 

```bash
python plot_cie.py
```

### 6. Backdoor Attention Head Ablation

Perform ablation studies on backdoor attention heads:

```bash
python backdoor_attention_ablation.py
```

### 7. Apply Backdoor Vector

Apply the computed backdoor vector to the model:

```bash
python backdoor_vector.py
```