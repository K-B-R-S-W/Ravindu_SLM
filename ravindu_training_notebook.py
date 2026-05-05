# ╔══════════════════════════════════════════════════════════════════╗
# ║     RAVINDU PERSONAL AI — QWEN 2.5 FINE-TUNING NOTEBOOK        ║
# ║     Model: Qwen/Qwen2.5-1.5B-Instruct + Unsloth + LoRA         ║
# ║     Optimized for: Colab Free T4 | Anti-hallucination           ║
# ║     Author: Built for Ravindu Sankalpa                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — INSTALL DEPENDENCIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run this first. Takes 2-3 minutes.
# WHY Unsloth? 2x faster training, 60% less VRAM than standard HF.

"""
!pip install unsloth
!pip install --upgrade --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft accelerate bitsandbytes datasets
!pip install rouge-score matplotlib seaborn
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

print("✅ All imports successful")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — CONFIGURATION (all settings in one place)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIG = {
    # ── Model ────────────────────────────────────────────────────
    # WHY Qwen2.5-1.5B? Small enough for free T4 VRAM (15GB),
    # smart enough for a personal assistant.
    # If you have Colab Pro, try Qwen2.5-3B-Instruct instead.
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct",

    # ── LoRA Parameters ──────────────────────────────────────────
    # WHY r=16? Rank 16 is the sweet spot for a dataset this size.
    # r=8 = less expressive, r=32 = more VRAM, risk of overfitting.
    "lora_r": 16,

    # WHY alpha=16? Keep alpha == r for stable training.
    # alpha controls the scale of LoRA updates: scale = alpha/r = 1.0
    "lora_alpha": 16,

    # WHY 0.05 dropout? Small dropout prevents overfitting on 2803 examples.
    # 0.0 = no regularization (risky), 0.1 = too much for small datasets.
    "lora_dropout": 0.05,

    # ── Target Layers ────────────────────────────────────────────
    # WHY these 7? Covers both attention (q,k,v,o) and FFN (gate,up,down).
    # This is the full standard set for chat fine-tuning.
    # Skipping embed_tokens and lm_head saves VRAM and reduces overfitting.
    "target_modules": [
        "q_proj",    # Query — what is this token looking for?
        "k_proj",    # Key — what does each token offer?
        "v_proj",    # Value — what information to extract
        "o_proj",    # Output — combine attention heads
        "gate_proj", # FFN gate — controls information flow (SiLU activation)
        "up_proj",   # FFN up — expands to higher dimension
        "down_proj", # FFN down — compresses back
    ],

    # ── Quantization ─────────────────────────────────────────────
    # WHY 4-bit? Reduces model from ~3GB to ~1GB VRAM.
    # Makes Qwen2.5-1.5B comfortably fit on T4 with room for training.
    "load_in_4bit": True,

    # ── Training ─────────────────────────────────────────────────
    # WHY 3 epochs? With 2803 examples: enough to learn, won't overfit.
    # More than 4 epochs on this dataset size = overfitting risk.
    "num_epochs": 3,

    # WHY batch=2, accumulation=4? Effective batch = 2x4 = 8.
    # Larger effective batch = more stable gradients.
    # Can't do batch=8 directly — T4 would OOM.
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,

    # WHY 2e-4? Standard LoRA learning rate. Too high (>5e-4) = unstable.
    # Too low (<1e-5) = doesn't learn. 2e-4 is the proven sweet spot.
    "learning_rate": 2e-4,

    # WHY cosine? Warms up slowly (avoids early instability),
    # then smoothly decays. Better than constant LR for fine-tuning.
    "lr_scheduler_type": "cosine",

    # WHY 10 warmup steps? Prevents large gradient updates at the start
    # when the LoRA weights are randomly initialized.
    "warmup_steps": 10,

    # WHY 1.0? Clips gradients to prevent exploding updates.
    # Standard value — don't change this.
    "max_grad_norm": 1.0,

    # WHY 2048? Max sequence length. Most examples are well under this.
    # Higher = more VRAM. 2048 covers even long coding answers.
    "max_seq_length": 2048,

    # WHY 0.1 weight decay? L2 regularization to prevent overfitting.
    # Penalizes large weights, keeps model general.
    "weight_decay": 0.1,

    # ── Evaluation ───────────────────────────────────────────────
    "eval_steps": 50,       # Evaluate every 50 steps
    "save_steps": 100,      # Save checkpoint every 100 steps
    "logging_steps": 10,    # Log metrics every 10 steps

    # ── Output ───────────────────────────────────────────────────
    "output_dir": "./ravindu_ai_checkpoints",
    "final_model_dir": "./ravindu_ai_final",
}

print("✅ Configuration loaded")
print(f"   Model: {CONFIG['model_name']}")
print(f"   LoRA rank: {CONFIG['lora_r']} | alpha: {CONFIG['lora_alpha']}")
print(f"   Epochs: {CONFIG['num_epochs']} | Effective batch: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"   Learning rate: {CONFIG['learning_rate']} ({CONFIG['lr_scheduler_type']} schedule)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — LOAD MODEL + TOKENIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY FastLanguageModel? Unsloth's optimized wrapper.
# Applies 4-bit quantization + prepares model for LoRA in one step.
# About 2-3 minutes to download and load.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=CONFIG["load_in_4bit"],
    dtype=None,  # Auto — uses bf16 if supported, else fp16
)

print(f"✅ Model loaded: {CONFIG['model_name']}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M total")

# Apply LoRA
# WHY get_peft_model? Injects trainable LoRA matrices into target layers.
# Everything else stays frozen — only LoRA weights update during training.
model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=CONFIG["target_modules"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",      # WHY none? Adding bias to LoRA layers rarely helps
    use_gradient_checkpointing="unsloth",  # WHY? Saves ~30% VRAM
    random_state=42,
    use_rslora=False, # WHY False? RSLoRA is for very large ranks (r>32)
)

# Count trainable vs frozen parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n✅ LoRA applied to {len(CONFIG['target_modules'])} layer types")
print(f"   Trainable parameters: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")
print(f"   Frozen parameters:    {(total-trainable)/1e6:.0f}M ({(total-trainable)/total*100:.2f}%)")
print(f"   💡 Only {trainable/total*100:.1f}% of weights update — this is why LoRA is efficient!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — LOAD + PREPARE DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Apply Qwen chat template to tokenizer
# WHY? Qwen has a specific format: <|im_start|>role\ncontent<|im_end|>
# The model was pretrained expecting this format — we must match it.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def format_example(example):
    """
    Convert each example to Qwen chat format.

    Input:  {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}
    Output: {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."}

    WHY format this way? The model was pretrained on this exact format.
    Using the wrong format = model doesn't know when to start/stop speaking.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Load dataset
print("Loading final_dataset.jsonl...")
raw_dataset = load_dataset("json", data_files="final_dataset.jsonl", split="train")
print(f"  Total examples: {len(raw_dataset)}")

# Format all examples
formatted = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

# Split into train/val
# WHY 95/5? With only 2803 examples, a larger val set wastes training data.
# 5% = ~140 examples — enough to detect overfitting.
split = formatted.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
val_dataset   = split["test"]

print(f"\n✅ Dataset prepared")
print(f"   Train: {len(train_dataset)} examples")
print(f"   Val:   {len(val_dataset)} examples")
print(f"\n   Sample formatted example (first 300 chars):")
print(f"   {train_dataset[0]['text'][:300]}...")

# Token length analysis
# WHY check this? If most examples are under 512 tokens, we can reduce
# max_seq_length and save significant VRAM.
print("\n   Analyzing token lengths...")
sample_lengths = []
for ex in train_dataset.select(range(min(200, len(train_dataset)))):
    tokens = tokenizer(ex["text"], return_tensors="pt")
    sample_lengths.append(tokens["input_ids"].shape[1])

print(f"   Token lengths — min:{min(sample_lengths)} | avg:{int(np.mean(sample_lengths))} | max:{max(sample_lengths)} | p95:{int(np.percentile(sample_lengths, 95))}")
print(f"   💡 95% of examples are under {int(np.percentile(sample_lengths, 95))} tokens")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — METRICS TRACKER (custom callback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY custom callback? HuggingFace logs metrics but doesn't store them
# in a format we can easily plot. This stores everything for visualization.

class MetricsTracker(TrainerCallback):
    """
    Tracks training metrics at every logging step.

    Metrics tracked:
    - train_loss:    How wrong on training data (want: steadily decreasing)
    - eval_loss:     How wrong on unseen val data (want: close to train_loss)
    - perplexity:    exp(eval_loss) — how surprised model is (want: < 10)
    - grad_norm:     Gradient size (want: stable, under 1.0)
    - learning_rate: LR schedule (want: warm up then decay)
    - loss_gap:      eval_loss - train_loss (want: small, <0.5 = healthy)

    Overfitting warning signs:
    - eval_loss starts RISING while train_loss keeps falling
    - loss_gap > 0.5 and growing
    - perplexity stops improving
    """
    def __init__(self):
        self.train_losses    = []
        self.eval_losses     = []
        self.perplexities    = []
        self.grad_norms      = []
        self.learning_rates  = []
        self.steps           = []
        self.eval_steps      = []
        self.best_eval_loss  = float("inf")
        self.patience        = 0
        self.MAX_PATIENCE    = 3  # early stopping after 3 bad evals

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(step)
        if "grad_norm" in logs:
            self.grad_norms.append(logs["grad_norm"])
        if "learning_rate" in logs:
            self.learning_rates.append(logs["learning_rate"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is not None:
            perplexity = math.exp(min(eval_loss, 20))  # cap to avoid overflow
            self.eval_losses.append(eval_loss)
            self.perplexities.append(perplexity)
            self.eval_steps.append(state.global_step)

            # Overfitting detection
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.patience = 0
                print(f"\n  ✅ New best eval loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
            else:
                self.patience += 1
                print(f"\n  ⚠️  Eval loss not improving ({self.patience}/{self.MAX_PATIENCE})")
                if self.patience >= self.MAX_PATIENCE:
                    print("  🛑 Early stopping triggered — preventing overfitting!")
                    control.should_training_stop = True

            # Loss gap check
            if self.train_losses:
                gap = eval_loss - self.train_losses[-1]
                if gap > 0.5:
                    print(f"  ⚠️  Loss gap = {gap:.3f} (>0.5) — watch for overfitting!")
                else:
                    print(f"  ✅ Loss gap = {gap:.3f} — healthy generalization")

    def summary(self):
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        if self.train_losses:
            print(f"  Final train loss:  {self.train_losses[-1]:.4f}")
        if self.eval_losses:
            print(f"  Best eval loss:    {self.best_eval_loss:.4f}")
            print(f"  Final perplexity:  {self.perplexities[-1]:.2f}")
            gap = self.eval_losses[-1] - self.train_losses[-1] if self.train_losses else 0
            print(f"  Final loss gap:    {gap:.4f}")
            if gap < 0.3:
                print("  ✅ Excellent generalization — no overfitting detected")
            elif gap < 0.5:
                print("  ✅ Good generalization — slight gap is normal")
            else:
                print("  ⚠️  Large gap — model may be overfitting")

metrics_tracker = MetricsTracker()
print("✅ Metrics tracker ready")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — TRAINING ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],

    # Batch + accumulation
    # WHY? Effective batch = 2 × 4 = 8. Stable training without OOM.
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],

    # Epochs
    # WHY 3? Just right for 2803 examples.
    # Formula: if val_loss stops improving before epoch 3, early stopping kicks in.
    num_train_epochs=CONFIG["num_epochs"],

    # Optimizer
    # WHY paged_adamw_32bit? Best for Colab T4:
    # - 32bit = more precise weight updates than 8bit
    # - paged = optimizer states spill to CPU RAM instead of crashing
    optim="paged_adamw_32bit",

    # Learning rate
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    warmup_steps=CONFIG["warmup_steps"],

    # Regularization
    # WHY weight_decay=0.1? L2 penalty prevents weights getting too large.
    # Key anti-overfitting measure alongside LoRA dropout.
    weight_decay=CONFIG["weight_decay"],

    # Gradient clipping
    # WHY 1.0? Prevents explosive gradient updates from corrupting LoRA weights.
    max_grad_norm=CONFIG["max_grad_norm"],

    # Precision
    # WHY fp16? T4 supports fp16. Use bf16=True if on A100/H100.
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    # Logging + eval
    logging_steps=CONFIG["logging_steps"],
    evaluation_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=2,  # Keep only 2 checkpoints to save disk space
    load_best_model_at_end=True,  # WHY? Auto-loads best checkpoint when done
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Reproducibility
    seed=42,
    report_to="none",  # Disable wandb/tensorboard — not needed for this project
)

print("✅ Training arguments configured")
print(f"   Optimizer: paged_adamw_32bit")
print(f"   Precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
print(f"   Early stopping: after {metrics_tracker.MAX_PATIENCE} non-improving evals")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — TRAINER SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY SFTTrainer? Built on top of HF Trainer, handles chat format
# and sequence packing automatically. Unsloth optimized.

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",

    # WHY max_seq_length here too? SFTTrainer packs sequences up to this length.
    # Packing = fills each batch slot with multiple short examples = faster training.
    max_seq_length=CONFIG["max_seq_length"],

    args=training_args,
    callbacks=[metrics_tracker],

    # WHY dataset_num_proc=2? Parallel tokenization = faster data prep.
    dataset_num_proc=2,

    # WHY packing=False? With 2803 examples some are long (code examples).
    # Packing can mix examples in confusing ways for small datasets.
    # Set True for larger datasets (>10k) to speed up training.
    packing=False,
)

print("✅ Trainer ready")
print(f"   Train steps per epoch: {len(train_dataset) // (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps'])}")
print(f"   Total training steps: ~{len(train_dataset) * CONFIG['num_epochs'] // (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps'])}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expected time on free T4: 45-90 minutes
# Watch for:
#   ✅ train_loss steadily decreasing
#   ✅ eval_loss staying close to train_loss
#   ⚠️  If eval_loss goes UP while train_loss goes DOWN = overfitting

print("🚀 Starting training...")
print("   Watch the loss gap (eval_loss - train_loss)")
print("   Healthy: gap < 0.3 | Warning: gap > 0.5\n")

trainer_stats = trainer.train()

metrics_tracker.summary()

print(f"\n   Training time: {trainer_stats.metrics['train_runtime'] / 60:.1f} minutes")
print(f"   Samples/second: {trainer_stats.metrics['train_samples_per_second']:.1f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 — VISUALIZE TRAINING METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_training_metrics(tracker):
    """
    Plots 5 key charts:
    1. Loss curves (train + val) — main overfitting indicator
    2. Perplexity — how confused the model is
    3. Gradient norm — training stability
    4. Learning rate schedule — confirm cosine decay is working
    5. Loss gap — generalization health over time
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Ravindu AI — Training Metrics", fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        "train": "#2563EB",
        "eval":  "#DC2626",
        "good":  "#16A34A",
        "warn":  "#D97706",
        "grad":  "#7C3AED",
        "lr":    "#0891B2",
        "gap":   "#DB2777",
    }

    # ── Chart 1: Loss Curves ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if tracker.steps and tracker.train_losses:
        ax1.plot(tracker.steps, tracker.train_losses,
                 color=colors["train"], label="Train Loss", linewidth=2, alpha=0.9)
    if tracker.eval_steps and tracker.eval_losses:
        ax1.plot(tracker.eval_steps, tracker.eval_losses,
                 color=colors["eval"], label="Val Loss",
                 linewidth=2, linestyle="--", marker="o", markersize=4)
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate("✅ Both should decrease\n⚠️ Gap > 0.5 = overfitting",
                 xy=(0.05, 0.05), xycoords="axes fraction",
                 fontsize=7, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # ── Chart 2: Perplexity ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if tracker.eval_steps and tracker.perplexities:
        ax2.plot(tracker.eval_steps, tracker.perplexities,
                 color=colors["eval"], linewidth=2, marker="s", markersize=4)
        ax2.axhline(y=5, color=colors["good"], linestyle="--", alpha=0.7, label="Excellent (<5)")
        ax2.axhline(y=10, color=colors["warn"], linestyle="--", alpha=0.7, label="Acceptable (<10)")
        ax2.legend(fontsize=8)
    ax2.set_title("Perplexity (Val)", fontweight="bold")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Perplexity = exp(loss)")
    ax2.grid(True, alpha=0.3)
    ax2.annotate("Lower = better\n<5: Excellent | <10: Good",
                 xy=(0.05, 0.85), xycoords="axes fraction",
                 fontsize=7, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # ── Chart 3: Gradient Norm ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if tracker.steps and tracker.grad_norms:
        steps_for_grads = tracker.steps[:len(tracker.grad_norms)]
        ax3.plot(steps_for_grads, tracker.grad_norms,
                 color=colors["grad"], linewidth=1.5, alpha=0.8)
        ax3.axhline(y=1.0, color=colors["warn"], linestyle="--",
                    alpha=0.8, label="Clip threshold (1.0)")
        ax3.legend(fontsize=8)
    ax3.set_title("Gradient Norm", fontweight="bold")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Grad Norm")
    ax3.grid(True, alpha=0.3)
    ax3.annotate("Should stay near/below 1.0\nSpikes = instability",
                 xy=(0.05, 0.85), xycoords="axes fraction",
                 fontsize=7, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # ── Chart 4: Learning Rate Schedule ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if tracker.steps and tracker.learning_rates:
        steps_for_lr = tracker.steps[:len(tracker.learning_rates)]
        ax4.plot(steps_for_lr, tracker.learning_rates,
                 color=colors["lr"], linewidth=2)
        ax4.fill_between(steps_for_lr, tracker.learning_rates,
                         alpha=0.15, color=colors["lr"])
    ax4.set_title("Learning Rate Schedule", fontweight="bold")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Learning Rate")
    ax4.grid(True, alpha=0.3)
    ax4.annotate("Cosine: warms up → peaks → decays\nSmooth decay = stable training",
                 xy=(0.05, 0.05), xycoords="axes fraction",
                 fontsize=7, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # ── Chart 5: Loss Gap (Overfitting Monitor) ──────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if tracker.eval_steps and tracker.eval_losses and tracker.train_losses:
        # Interpolate train loss at eval steps
        gaps = []
        for i, eval_step in enumerate(tracker.eval_steps):
            closest_idx = min(range(len(tracker.steps)),
                              key=lambda j: abs(tracker.steps[j] - eval_step))
            gap = tracker.eval_losses[i] - tracker.train_losses[closest_idx]
            gaps.append(gap)

        colors_bar = [colors["good"] if g < 0.3
                      else colors["warn"] if g < 0.5
                      else colors["eval"] for g in gaps]
        ax5.bar(tracker.eval_steps, gaps, color=colors_bar, alpha=0.8, width=8)
        ax5.axhline(y=0.3, color=colors["good"], linestyle="--",
                    alpha=0.7, label="Healthy (<0.3)")
        ax5.axhline(y=0.5, color=colors["warn"], linestyle="--",
                    alpha=0.7, label="Warning (>0.5)")
        ax5.legend(fontsize=8)
    ax5.set_title("Loss Gap (Overfitting Monitor)", fontweight="bold")
    ax5.set_xlabel("Steps")
    ax5.set_ylabel("Val Loss − Train Loss")
    ax5.grid(True, alpha=0.3)

    # ── Chart 6: Training Summary Table ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_data = []
    if tracker.train_losses:
        summary_data.append(["Final Train Loss", f"{tracker.train_losses[-1]:.4f}", "↓ lower better"])
    if tracker.eval_losses:
        summary_data.append(["Best Val Loss", f"{tracker.best_eval_loss:.4f}", "↓ lower better"])
        summary_data.append(["Final Perplexity", f"{tracker.perplexities[-1]:.2f}", "<5 excellent"])
    if tracker.grad_norms:
        summary_data.append(["Avg Grad Norm", f"{np.mean(tracker.grad_norms):.3f}", "<1.0 stable"])
    if summary_data:
        table = ax6.table(
            cellText=summary_data,
            colLabels=["Metric", "Value", "Target"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0.2, 1, 0.7]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1E293B")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F1F5F9")
    ax6.set_title("Summary", fontweight="bold")

    plt.savefig("training_metrics.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    print("✅ Charts saved to training_metrics.png")

plot_training_metrics(metrics_tracker)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — ROUGE SCORE EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY ROUGE? Measures overlap between model output and reference answer.
# ROUGE-1: unigram overlap | ROUGE-2: bigram overlap | ROUGE-L: longest match
# For a personal AI assistant ROUGE-L > 0.3 is good, >0.5 is excellent.

def evaluate_rouge(model, tokenizer, val_dataset, n_samples=30):
    """
    Runs ROUGE evaluation on n_samples from validation set.
    Generates model outputs and compares to reference outputs.
    """
    FastLanguageModel.for_inference(model)  # Enable faster inference mode

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    print(f"Running ROUGE evaluation on {n_samples} samples...")

    sample_indices = np.random.choice(len(val_dataset), min(n_samples, len(val_dataset)), replace=False)

    for idx in sample_indices:
        example = val_dataset[int(idx)]
        full_text = example["text"]

        # Split at last assistant turn to get reference
        split_token = "<|im_start|>assistant"
        parts = full_text.rsplit(split_token, 1)
        if len(parts) != 2:
            continue

        prompt = parts[0] + split_token + "\n"
        reference = parts[1].replace("<|im_end|>", "").strip()

        # Tokenize prompt only
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,        # WHY 0.7? Balance creativity vs accuracy
                top_p=0.9,              # WHY 0.9? Nucleus sampling — avoids low-prob tokens
                repetition_penalty=1.1, # WHY 1.1? Penalizes repeating same phrases
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if generated and reference:
            result = scorer.score(reference, generated)
            for key in scores:
                scores[key].append(result[key].fmeasure)

    # Average scores
    avg_scores = {k: np.mean(v) for k, v in scores.items() if v}

    print("\n✅ ROUGE Evaluation Results:")
    print(f"   ROUGE-1: {avg_scores.get('rouge1', 0):.4f}  (unigram overlap)")
    print(f"   ROUGE-2: {avg_scores.get('rouge2', 0):.4f}  (bigram overlap)")
    print(f"   ROUGE-L: {avg_scores.get('rougeL', 0):.4f}  (longest common subsequence)")
    print()
    rougeL = avg_scores.get('rougeL', 0)
    if rougeL > 0.5:
        print("   ✅ Excellent! Model outputs closely match expected responses.")
    elif rougeL > 0.3:
        print("   ✅ Good. Model is generating relevant responses.")
    elif rougeL > 0.15:
        print("   ⚠️  Moderate. Model understands the domain but paraphrases a lot.")
    else:
        print("   ⚠️  Low. Model may need more training or data.")

    # Plot ROUGE scores
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_names = list(avg_scores.keys())
    metric_values = list(avg_scores.values())
    bar_colors = ["#16A34A" if v > 0.5 else "#2563EB" if v > 0.3 else "#D97706"
                  for v in metric_values]
    bars = ax.bar(metrics_names, metric_values, color=bar_colors, alpha=0.85, width=0.5)
    ax.axhline(y=0.5, color="#16A34A", linestyle="--", alpha=0.7, label="Excellent (>0.5)")
    ax.axhline(y=0.3, color="#2563EB", linestyle="--", alpha=0.7, label="Good (>0.3)")
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("ROUGE Scores — Model Evaluation", fontweight="bold", fontsize=13)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("rouge_scores.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ ROUGE chart saved to rouge_scores.png")

    return avg_scores

rouge_scores = evaluate_rouge(model, tokenizer, val_dataset)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — QUALITATIVE TESTING (talk to your model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY qualitative testing? ROUGE only measures text overlap.
# A model can have low ROUGE but give great answers (just worded differently).
# Always test with real questions as the final check.

def chat(prompt, max_new_tokens=300, temperature=0.7):
    """
    Send a message to your fine-tuned Ravindu AI.

    Parameters:
    - max_new_tokens: How long the response can be (300 = ~200 words)
    - temperature: 0.1-0.3 for factual, 0.7-0.9 for casual/creative
    """
    FastLanguageModel.for_inference(model)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Ravindu Sankalpa's personal AI assistant. "
                "You know Ravindu well and help him with daily tasks, coding, AI/ML questions, "
                "career advice, and casual conversation. Be friendly, casual, and direct."
            )
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"You: {prompt}")
    print(f"AI:  {response}")
    print()
    return response

# Run test questions — these cover all the key areas
print("=" * 55)
print("QUALITATIVE TESTING — Talk to Ravindu AI")
print("=" * 55)

test_questions = [
    # Personal knowledge test
    ("PERSONAL",            "what is CrimeGuard"),
    ("PERSONAL",            "tell me about your SLT internship"),
    # Casual chat test
    ("CASUAL",              "hey bro what's up"),
    ("CASUAL",              "im so tired today"),
    # Technical test
    ("TECHNICAL",           "what is RAG and how does it work"),
    ("TECHNICAL",           "how do i load a csv in pandas"),
    # Sri Lanka context test
    ("SL CONTEXT",          "what is the salary for fresh AI graduates in sri lanka"),
    # Hallucination test — model should say it doesn't know, not make things up
    ("HALLUCINATION CHECK", "what did you eat for breakfast today"),
]

for category, question in test_questions:
    print(f"[{category}]")
    chat(question)
    print("-" * 40)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — SAVE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY save LoRA separately first? Much smaller file (MBs vs GBs).
# Merge only when you need the full standalone model for deployment.

print("Saving LoRA adapter (fast, small)...")
model.save_pretrained(CONFIG["final_model_dir"])
tokenizer.save_pretrained(CONFIG["final_model_dir"])
print(f"✅ LoRA adapter saved to {CONFIG['final_model_dir']}")

# Merge LoRA into base model for deployment using Unsloth's method
# WHY save_pretrained_merged? Bypasses broken HuggingFace weight
# reversion that causes NotImplementedError with newer transformers.
print("\nMerging LoRA into base model (for deployment)...")
model.save_pretrained_merged(
    CONFIG["final_model_dir"] + "_merged",
    tokenizer,
    save_method="merged_16bit"  # full precision, best quality
)
print(f"✅ Merged model saved to {CONFIG['final_model_dir']}_merged")

# Optional: save in GGUF format for Ollama/llama.cpp local deployment
# Uncomment if you want to run it locally on CPU:
# model.save_pretrained_gguf(
#     CONFIG["final_model_dir"] + "_gguf",
#     tokenizer,
#     quantization_method="q4_k_m"  # 4-bit quantized, good quality/size tradeoff
# )
# print("✅ GGUF model saved for local CPU inference")

print("\n" + "="*55)
print("🎉 TRAINING COMPLETE!")
print("="*55)
print(f"""
Your model is saved and ready.

Files created:
  {CONFIG['final_model_dir']}/          <- LoRA adapter only (small)
  {CONFIG['final_model_dir']}_merged/   <- Full merged model
  training_metrics.png                  <- Training charts
  rouge_scores.png                      <- ROUGE evaluation chart

To use your model later:
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      "{CONFIG['final_model_dir']}_merged"
  )
""")
