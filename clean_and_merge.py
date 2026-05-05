"""
Dataset Merging Script for Ravindu's Personal AI
=================================================
What this does:
1. Loads ravindu_personal_merged.jsonl  - your cleaned personal dataset
2. Downloads FuseChat (2000 samples)    - sampled evenly across all 95k
3. Merges everything + saves final_dataset.jsonl

HOW TO USE IN GOOGLE COLAB:
    Upload ravindu_personal_merged.jsonl + this script
    !pip install datasets
    !python clean_and_merge.py

Output: final_dataset.jsonl (~2800 examples, ready for Unsloth)
"""

import json
import random
random.seed(42)

SYSTEM_PROMPT = (
    "You are Ravindu Sankalpa's personal AI assistant. "
    "You know Ravindu well and help him with daily tasks, coding, AI/ML questions, "
    "career advice, and casual conversation. Be friendly, casual, and direct."
)

GENERAL_SYSTEM = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Answer clearly and naturally."
)

# ─────────────────────────────────────────
# STEP 1 — Load personal data
# ─────────────────────────────────────────
print("=" * 52)
print("STEP 1: Loading ravindu_personal_merged.jsonl")
print("=" * 52)

personal_data = []

try:
    with open("ravindu_personal_merged.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                # Format A: instruction/output
                if "instruction" in obj and "output" in obj:
                    personal_data.append({
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": obj["instruction"]},
                            {"role": "assistant", "content": obj["output"]}
                        ]
                    })

                # Format B: messages
                elif "messages" in obj:
                    msgs = obj["messages"]
                    has_user = any(m.get("role") == "user" for m in msgs)
                    has_asst = any(m.get("role") == "assistant" for m in msgs)
                    if has_user and has_asst:
                        if not any(m.get("role") == "system" for m in msgs):
                            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
                        personal_data.append({"messages": msgs})

            except Exception:
                pass

    print(f"  Loaded: {len(personal_data)} examples")

except FileNotFoundError:
    print("  ERROR: ravindu_personal_merged.jsonl not found!")
    print("  Run merge_personal.py first.")
    exit(1)

# ─────────────────────────────────────────
# STEP 2 — Download FuseChat evenly sampled
# ─────────────────────────────────────────
print("\n" + "=" * 52)
print("STEP 2: Downloading FuseChat (evenly sampled)")
print("=" * 52)

fusechat_data = []
TOTAL_FUSECHAT = 2000  # total examples to take

try:
    from datasets import load_dataset

    print("  Loading full FuseChat-Mixture index (this takes a moment)...")
    full_dataset = load_dataset(
        "FuseAI/FuseChat-Mixture",
        split="train",
        trust_remote_code=True
    )

    total_size = len(full_dataset)
    print(f"  Full dataset size: {total_size} examples")
    print(f"  Sampling {TOTAL_FUSECHAT} evenly spread across all {total_size}...")

    # Evenly spaced indices across the full 95k
    # This guarantees we hit all 9 categories not just the first one
    step = total_size // TOTAL_FUSECHAT
    sampled_indices = list(range(0, total_size, step))[:TOTAL_FUSECHAT]

    print(f"  Sampling every {step}th example")
    print(f"  Index range: {sampled_indices[0]} to {sampled_indices[-1]}")

    sampled = full_dataset.select(sampled_indices)

    for row in sampled:
        # Handle both FuseChat formats
        if "messages" in row:
            msgs = list(row["messages"])
        elif "conversations" in row:
            msgs = []
            for turn in row["conversations"]:
                role = "user" if turn.get("from", "") == "human" else "assistant"
                msgs.append({"role": role, "content": turn.get("value", "")})
        else:
            continue

        has_user = any(m.get("role") == "user" for m in msgs)
        has_asst = any(m.get("role") == "assistant" for m in msgs)

        if has_user and has_asst:
            if not any(m.get("role") == "system" for m in msgs):
                msgs = [{"role": "system", "content": GENERAL_SYSTEM}] + msgs
            fusechat_data.append({"messages": msgs})

    print(f"  Loaded: {len(fusechat_data)} examples from FuseChat")

except ImportError:
    print("  ERROR: datasets not installed. Run: pip install datasets")

except Exception as e:
    print(f"  ERROR loading FuseChat: {e}")
    print("  Skipping FuseChat — continuing with personal data only...")

# ─────────────────────────────────────────
# STEP 3 — Merge + Shuffle + Save
# ─────────────────────────────────────────
print("\n" + "=" * 52)
print("STEP 3: Merging + Shuffling + Saving")
print("=" * 52)

all_data = personal_data + fusechat_data

print(f"  ravindu_personal_merged : {len(personal_data)}")
print(f"  FuseChat (even sample)  : {len(fusechat_data)}")
print(f"  {'─'*36}")
print(f"  TOTAL                   : {len(all_data)}")

random.shuffle(all_data)

output_path = "final_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in all_data:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"\n  Saved to: {output_path}")

# ─────────────────────────────────────────
# STEP 4 — Validate
# ─────────────────────────────────────────
print("\n" + "=" * 52)
print("STEP 4: Validating final_dataset.jsonl")
print("=" * 52)

valid = 0
invalid = 0

with open(output_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            has_user = any(m.get("role") == "user" for m in msgs)
            has_asst = any(m.get("role") == "assistant" for m in msgs)
            if has_user and has_asst:
                valid += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1

print(f"  Valid:   {valid}")
print(f"  Invalid: {invalid}")

if invalid == 0:
    print("\n  ALL GOOD! Your dataset is ready for Unsloth training.")
else:
    print(f"\n  WARNING: {invalid} invalid examples — check your input files.")

print("\n" + "=" * 52)
print("DONE!")
print("=" * 52)
print(f"""
Final dataset : {output_path}
Total examples: {valid}

Breakdown:
  Personal (Ravindu) : {len(personal_data)} ({len(personal_data)*100//max(valid,1)}%)
  General (FuseChat) : {len(fusechat_data)} ({len(fusechat_data)*100//max(valid,1)}%)

Load in Colab for Unsloth training:

    from datasets import load_dataset
    dataset = load_dataset("json", data_files="final_dataset.jsonl", split="train")
    dataset = dataset.train_test_split(test_size=0.05)
    train_data = dataset["train"]
    val_data   = dataset["test"]
    print(f"Train: {{len(train_data)}} | Val: {{len(val_data)}}")
""")
