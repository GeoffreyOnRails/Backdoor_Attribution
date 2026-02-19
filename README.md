# Backdoor_Attribution

This repository contains the code needed to reproduce the paper [BACKDOOR ATTRIBUTION: ELUCIDATING AND CONTROLLING BACKDOORS IN LANGUAGE MODELS](https://arxiv.org/pdf/2509.21761) and its general performance evaluation extension, as described in TODO.

## Setup Instructions

### 1. Setup Environment

Install the required dependencies using : 

```bash
uv sync
```

### 2. Create a backdoored model

Run the following command to inject a backdoor into the model:

```bash
CUDA_VISIBLE_DEVICES=0 python backdoor_sft.py --task_name=alpaca_begin --model_family=qwen2.5-7b
```

This will create a folder containing the LORA weight for the backdoored model in model_weight/alpaca_begin/qwen2.5-7b/attn_mlp

If you want to reproduce for llama2-7b, you first have to request access to the llama 2 family of models (https://huggingface.co/meta-llama/Llama-2-7b-hf), and then log in to huggingface using the following command : 

```
hf auth login
```

Then, follow the instructions to create and fill a token.

## Training Backdoor Probes

Train the classifier to train probes able to detect backdoor triggers:

```bash
python train_classifier.py
```

This is not required for the next steps

## Backdoor Attention Heads

### 1. Backdoor Attention Attribution

Calculate attention-based importance estimation (CIE) for backdoor analysis :

```bash
python calculate_cie.py --task_names alpaca_begin --model_families qwen2.5-7b
```

or

```bash
python calculate_cie.py --task_names alpaca_begin --model_families llama2-7b --use_flash_attn
```

Use the `--help` flag for more details:

```bash
python calculate_cie.py --help
```

If you want to reproduce the inter-layer classification accuracy graphs available in the paper BACKDOOR ATTRIBUTION: ELUCIDATING AND CONTROLLING BACKDOORS IN LANGUAGE MODELS, you can run :  

```bash
python plot_cie.py --model_family qwen2.5-7b --task_name alpaca_begin
```

### 2. Backdoor Attention Head Ablation

Perform ablation studies on backdoor attention heads :

```bash
python backdoor_attention_ablation.py --task_names alpaca_begin --model_families qwen2.5-7b --dataset eval
```

or

```bash
python backdoor_attention_ablation.py --task_names alpaca_begin --model_families llama2-7b --dataset eval --use_flash_attn
```

Use the `--help` flag for more details on the available options:

```bash
python backdoor_attention_ablation.py --help
```

### 3. Performance evaluation 

First, you need to add your Gemini key : 

```bash
export GEMINI_API_KEY="your-api-key"
```

Then, run the following command to evaluate the performance of the backdoored model:

```bash
python classify_ablation_results.py --input_dir=results/attn_ablation/alpaca_begin/qwen2.5-7b/attn_mlp/result_sample_eval_256/ --output_csv=results/evaluations/qwen2.5-7b/classification_results --temperature 0
```

You can compare different runs (either manual classification or from Gemini) using the following command : 

```bash
python compare_runs.py --run1 classification_results_r1_commented.csv --run2 classification_results_r2_commented.csv --run3 classification_results_r3_commented.csv --output compare_runs.csv --include_comments --add_output_from_folder=results/attn_ablation/alpaca_begin/qwen2.5-7b/attn_mlp/result_sample_eval_256
```

More details about the scripts' available options (e.g limiting the number of samples, running in dry-run mode, etc.) can be found using the `--help` flag : 

```bash
python compare_runs.py --help
python classify_ablation_results.py --help
```

## Backdoor Vectors

Apply the computed backdoor vector to the model:

```bash
python backdoor_vector.py
```


## Changing Base Model Paths

Edit the configuration file and set the paths to your model files in util.py:

```python
llama_model_path = "your/path/to/llama_model"
qwen_model_path = "your/path/to/qwen_model"
```