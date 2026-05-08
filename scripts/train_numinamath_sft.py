#!/usr/bin/env python3
"""
NuminaMath SFT Training Script

Train a LoRA adapter on the NuminaMath conversational dataset using SFTTrainer.
Usage: python scripts/train_numinamath_sft.py
"""

import os
import sys
import logging
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    from transformers import logging as hf_logging
except ImportError:
    from transformers.utils import logging as hf_logging

from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

try:
    from bert_score import BERTScorer
except ImportError:
    BERTScorer = None

# ============================================================================
# Configuration and Constants
# ============================================================================

def resolve_project_root() -> Path:
    """Return the repository root whether the script runs from repo root or scripts/."""
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd
    if (cwd.parent / "pyproject.toml").exists():
        return cwd.parent
    return cwd


PROJECT_ROOT = resolve_project_root()
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))

DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DATASET_DIR = DATASETS_DIR / "processed" / "numinamath_sft" / "train_990_seed_42"

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-0.6B-SFT"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft-numinamath-processed"
LOGGING_DIR = PROJECT_ROOT / "logs"

# Dataset subset configuration
USE_SUBSET = True
TRAIN_SUBSET_SIZE = 10000
TEST_SUBSET_SIZE = 200

# Training configuration
SMOKE_RUN = False  # Set to False for full training
NUM_EPOCHS = 3
RANK = 4

# Evaluation configuration
N_EVAL_EXAMPLES = 1
EVAL_SEED = 123
MAX_NEW_TOKENS = 512  # Reduced from 2048 for faster evaluation
TEST_METRIC_SUBSET_SIZE = 200

# BERTScore configuration
ROBERTA_MODEL_DIR = PROJECT_ROOT / "models" / "roberta-large"

# ============================================================================
# Utility Functions
# ============================================================================

@contextmanager
def suppress_transformers_logging():
    """Context manager to suppress transformers library logging."""
    old_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(old_verbosity)


def count_params(model):
    """Count total and trainable parameters in a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def extract_assistant_completion(completion):
    """Extract the assistant content from a conversational completion record."""
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return str(completion)


def parse_completion(completion_text):
    """Parse <think> and <answer> tags from generated text (strict parsing)."""
    reasoning = ""
    final_answer = ""
    
    if "<think>" in completion_text and "</think>" in completion_text:
        start_idx = completion_text.find("<think>") + len("<think>")
        end_idx = completion_text.find("</think>")
        reasoning = completion_text[start_idx:end_idx].strip()
    
    if "<answer>" in completion_text and "</answer>" in completion_text:
        answer_start = completion_text.find("<answer>") + len("<answer>")
        answer_end = completion_text.find("</answer>")
        final_answer = completion_text[answer_start:answer_end].strip() or "no answer tags"
    else:
        final_answer = "no answer tags"
    
    return reasoning, final_answer


def extract_original_problem(sample):
    """Extract the original problem from a sample's prompt messages."""
    prompt_messages = sample["prompt"]
    if isinstance(prompt_messages, list) and prompt_messages:
        return prompt_messages[0].get("content", "")
    return str(prompt_messages)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"CACHE_ROOT: {CACHE_ROOT}")
    print(f"PROCESSED_DATASET_DIR: {PROCESSED_DATASET_DIR}")
    print()
    
    # ========================================================================
    # 1. Load Dataset
    # ========================================================================
    print("Loading processed dataset...")
    if not PROCESSED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATASET_DIR}. "
            f"Run NumiNaMath_Preprocess_for_SFT.ipynb first."
        )
    
    processed_dataset = load_from_disk(str(PROCESSED_DATASET_DIR))
    train_dataset = cast(Dataset, processed_dataset["train"])
    test_dataset = cast(Dataset, processed_dataset["test"])
    
    # Apply subset selection if enabled
    if USE_SUBSET:
        n_train = min(TRAIN_SUBSET_SIZE, len(train_dataset))
        n_test = min(TEST_SUBSET_SIZE, len(test_dataset))
        train_dataset = train_dataset.select(range(n_train))
        test_dataset = test_dataset.select(range(n_test))
        print(f"Using subset: train={n_train}, test={n_test}")
    else:
        print(f"Using full dataset: train={len(train_dataset)}, test={len(test_dataset)}")
    
    # Validate conversational format
    required_keys = {"prompt", "completion"}
    first_train_sample = train_dataset[0]
    first_test_sample = test_dataset[0]
    
    assert required_keys.issubset(first_train_sample.keys())
    assert required_keys.issubset(first_test_sample.keys())
    assert isinstance(first_train_sample["prompt"], list) and first_train_sample["prompt"]
    assert isinstance(first_train_sample["completion"], list) and first_train_sample["completion"]
    assert first_train_sample["prompt"][0]["role"] == "user"
    assert first_train_sample["completion"][0]["role"] == "assistant"
    assert "<think>" in first_train_sample["completion"][0]["content"]
    assert "<answer>" in first_train_sample["completion"][0]["content"]
    
    print("Verified conversational prompt/completion format with tags.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print()
    
    # ========================================================================
    # 2. Load Model and Tokenizer
    # ========================================================================
    print("Loading model and tokenizer...")
    if not any(MODEL_DIR.iterdir()):
        print(f"{MODEL_DIR} is empty -> downloading model/tokenizer from Hub...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer.save_pretrained(str(MODEL_DIR))
        model.save_pretrained(str(MODEL_DIR))
    else:
        print(f"{MODEL_DIR} is not empty -> loading model/tokenizer from disk...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = model
    full_trainable, full_total = count_params(base_model)
    print(f"Model: {MODEL_ID}, full FT trainable: {full_trainable:,}")
    print()
    
    # ========================================================================
    # 3. Wrap Model with LoRA
    # ========================================================================
    print("Wrapping model with LoRA...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=RANK,
        lora_alpha=2 * RANK,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    lora_model = cast(PeftModel, get_peft_model(base_model, peft_config))
    
    lora_trainable, lora_total = count_params(lora_model)
    reduction_vs_full_ft = 100 * (1 - (lora_trainable / full_trainable))
    trainable_pct_of_total = 100 * (lora_trainable / lora_total)
    
    print(f"LoRA trainable: {lora_trainable:,}")
    print(f"Reduction: {reduction_vs_full_ft:.2f}%")
    print(f"LoRA trainable %: {trainable_pct_of_total:.4f}%")
    print()
    
    # ========================================================================
    # 4. Create SFTTrainer
    # ========================================================================
    print("Creating SFTTrainer...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    
    sft_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=NUM_EPOCHS,
        logging_strategy="steps",
        logging_dir=str(LOGGING_DIR),
        logging_steps=25,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        eval_steps=125,
        save_steps=625,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=8,
        completion_only_loss=True,
    )
    
    trainer = SFTTrainer(
        model=cast(PeftModel, lora_model),
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    print("Trainer created successfully.")
    print()
    
    # ========================================================================
    # 5. Train
    # ========================================================================
    print("Starting training...")
    trainer.train()
    print("Training completed.")
    print()
    
    model_for_eval = trainer.model
    model_for_eval.eval()
    
    # ========================================================================
    # 6. Save Trained Adapter
    # ========================================================================
    print("Saving trained adapter...")
    SAVE_DIR = OUTPUT_DIR / "final_adapter"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    print(f"Saved LoRA adapter and tokenizer to {SAVE_DIR}")
    print()
    
    # ========================================================================
    # 7. Generate Example Evaluations
    # ========================================================================
    print("Generating example evaluations...")
    rng = np.random.default_rng(EVAL_SEED)
    eval_indices = rng.choice(
        len(test_dataset),
        size=min(N_EVAL_EXAMPLES, len(test_dataset)),
        replace=False,
    ).tolist()
    
    for i, idx in enumerate(eval_indices):
        sample = test_dataset[idx]
        original_problem = extract_original_problem(sample)
        prompt_messages = sample["prompt"]
        ground_truth_completion = extract_assistant_completion(sample["completion"])
        
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_for_eval.device)
        
        with torch.no_grad():
            generated = model_for_eval.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[0][prompt_length:]
        predicted_completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        gt_reasoning, gt_answer = parse_completion(ground_truth_completion)
        pred_reasoning, pred_answer = parse_completion(predicted_completion)
        
        print("=" * 80)
        print(f"Example {i+1} / {N_EVAL_EXAMPLES}")
        print("Original problem:")
        print(original_problem)
        print()
        print("Ground truth answer:")
        print(gt_answer or "[empty]")
        print()
        print("Predicted answer:")
        print(pred_answer or "[empty]")
        print()
    
    # ========================================================================
    # 8. Full-Test BERTScore Metrics (if BERTScore available)
    # ========================================================================
    if BERTScorer is not None:
        print("Computing full-test BERTScore metrics...")
        
        # Setup BERTScore model with local caching
        ROBERTA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with suppress_transformers_logging():
                with redirect_stderr(StringIO()):
                    # Try to load from local cache first
                    if (ROBERTA_MODEL_DIR / "config.json").exists():
                        scorer = BERTScorer(
                            lang="en",
                            model_type=str(ROBERTA_MODEL_DIR),
                            num_layers=17,
                            rescale_with_baseline=False,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                    else:
                        # Download and cache
                        import transformers
                        roberta = transformers.AutoModel.from_pretrained(
                            "roberta-large",
                            cache_dir=str(ROBERTA_MODEL_DIR),
                        )
                        roberta.save_pretrained(str(ROBERTA_MODEL_DIR))
                        scorer = BERTScorer(
                            lang="en",
                            model_type=str(ROBERTA_MODEL_DIR),
                            num_layers=17,
                            rescale_with_baseline=False,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
        except Exception as e:
            print(f"Warning: Could not initialize BERTScorer: {e}")
            print("Skipping BERTScore metrics.")
            return
        
        # Generate predictions on metric subset
        metric_dataset = test_dataset.select(
            range(min(TEST_METRIC_SUBSET_SIZE, len(test_dataset)))
        )
        predicted_answers = []
        ground_truth_answers = []
        example_records = []
        
        for idx in tqdm(range(len(metric_dataset)), desc="Predicting test answers"):
            sample = metric_dataset[idx]
            prompt_messages = sample["prompt"]
            ground_truth_completion = extract_assistant_completion(sample["completion"])
            
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model_for_eval.device)
            
            with torch.no_grad():
                generated = model_for_eval.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced for metrics loop
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            prompt_length = inputs["input_ids"].shape[1]
            generated_tokens = generated[0][prompt_length:]
            predicted_completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            _, ground_truth_answer = parse_completion(ground_truth_completion)
            _, predicted_answer = parse_completion(predicted_completion)
            
            ground_truth_answers.append(ground_truth_answer)
            predicted_answers.append(predicted_answer)
            example_records.append({
                "index": idx,
                "problem": prompt_messages[0].get("content", "") if prompt_messages else "",
                "predicted_answer": predicted_answer,
                "ground_truth_answer": ground_truth_answer,
            })
        
        # Compute BERTScore
        try:
            _, _, f1_scores = scorer.score(
                predicted_answers,
                ground_truth_answers,
                batch_size=2,
            )
            f1_scores = f1_scores.cpu().numpy() if hasattr(f1_scores, "cpu") else f1_scores
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return
        
        # Report metrics
        mean_f1 = float(np.mean(f1_scores))
        print()
        print(f"Metric test subset size: {len(metric_dataset)}")
        print(f"Mean BERTScore F1: {mean_f1:.4f}")
        
        # Find and print worst 3 cases
        worst_indices = np.argsort(f1_scores)[:3]
        print("Worst 3 BERTScore F1 cases:")
        
        for rank, worst_idx in enumerate(worst_indices, 1):
            example = example_records[worst_idx]
            print("-" * 80)
            print(f"Index: {example['index']}")
            print(f"BERTScore F1: {f1_scores[worst_idx]:.4f}")
            print("Problem:")
            print(example["problem"])
            print()
            print("Ground truth answer:")
            print(example["ground_truth_answer"])
            print()
            print("Predicted answer:")
            print(example["predicted_answer"])
            print()
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()