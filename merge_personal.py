import json
import os
import re
import glob

JSONL_FILES = [
    "batch_01.jsonl",
    "batch_02.jsonl",
    "batch_03.jsonl",
    "batch_04.jsonl",
]

OUTPUT_FILE = "ravindu_personal.jsonl"

AUTO_DISCOVER = True

if AUTO_DISCOVER:
    all_files = glob.glob("*.jsonl")
    all_files = [f for f in all_files if f != OUTPUT_FILE and "checkpoint" not in f and "final_dataset" not in f and "fusechat" not in f.lower()]
    print(f"\nAuto-discovered {len(all_files)} files:")
else:
    all_files = [f for f in JSONL_FILES if os.path.exists(f)]
    missing = [f for f in JSONL_FILES if not os.path.exists(f)]
    print(f"\nUsing {len(all_files)} files:")
    if missing:
        print(f"  Missing (skipped): {missing}")

for f in sorted(all_files):
    size = os.path.getsize(f) / 1024
    print(f"  - {f} ({size:.1f} KB)")

raw_examples = []
file_stats = {}

for filepath in sorted(all_files):
    loaded = 0
    skipped = 0
    parse_errors = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("'") and line.endswith("'"):
                line = line[1:-1]

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    fixed = line.replace('\\"', '"').replace("\\n", "\n")
                    obj = json.loads(fixed)
                except Exception:
                    parse_errors += 1
                    continue

            example = None

            if "instruction" in obj and "output" in obj:
                example = {
                    "instruction": str(obj["instruction"]).strip(),
                    "output": str(obj["output"]).strip()
                }

            elif "messages" in obj:
                msgs = obj["messages"]
                user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
                if user and assistant:
                    example = {
                        "instruction": str(user).strip(),
                        "output": str(assistant).strip()
                    }

            elif "conversations" in obj:
                convs = obj["conversations"]
                user = next((c["value"] for c in convs if c.get("from") in ["human", "user"]), None)
                assistant = next((c["value"] for c in convs if c.get("from") in ["gpt", "assistant"]), None)
                if user and assistant:
                    example = {
                        "instruction": str(user).strip(),
                        "output": str(assistant).strip()
                    }

            elif "prompt" in obj and "response" in obj:
                example = {
                    "instruction": str(obj["prompt"]).strip(),
                    "output": str(obj["response"]).strip()
                }

            if example:
                example["_source"] = filepath
                raw_examples.append(example)
                loaded += 1
            else:
                skipped += 1

    file_stats[filepath] = {"loaded": loaded, "skipped": skipped, "errors": parse_errors}
    print(f"  {filepath}: {loaded} loaded | {skipped} format-skipped | {parse_errors} parse errors")

print(f"\n  Total raw examples: {len(raw_examples)}")


print(f"\n{'='*55}")
print("STEP 3: Cleaning examples")
print("="*55)

def clean_text(text):
    """Clean a single text field."""
    text = text.strip()

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r' {3,}', ' ', text)

    text = text.replace('\x00', '')

    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    return text.strip()


cleaned_examples = []
cleaning_stats = {
    "too_short_instruction": 0,
    "too_short_output": 0,
    "too_long_output": 0,
    "empty_fields": 0,
    "cleaned_ok": 0,
}

MIN_INSTRUCTION_LEN = 3
MIN_OUTPUT_LEN = 10
MAX_OUTPUT_LEN = 4000

for ex in raw_examples:
    instruction = clean_text(ex["instruction"])
    output = clean_text(ex["output"])

    if not instruction or not output:
        cleaning_stats["empty_fields"] += 1
        continue

    if len(instruction) < MIN_INSTRUCTION_LEN:
        cleaning_stats["too_short_instruction"] += 1
        continue

    if len(output) < MIN_OUTPUT_LEN:
        cleaning_stats["too_short_output"] += 1
        continue

    if len(output) > MAX_OUTPUT_LEN:
        output = output[:MAX_OUTPUT_LEN].rsplit(' ', 1)[0] + "..."
        cleaning_stats["too_long_output"] += 1

    cleaned_examples.append({
        "instruction": instruction,
        "output": output,
        "_source": ex["_source"]
    })
    cleaning_stats["cleaned_ok"] += 1

print(f"  Passed cleaning:         {cleaning_stats['cleaned_ok']}")
print(f"  Removed - empty fields:  {cleaning_stats['empty_fields']}")
print(f"  Removed - short input:   {cleaning_stats['too_short_instruction']}")
print(f"  Removed - short output:  {cleaning_stats['too_short_output']}")
print(f"  Truncated - long output: {cleaning_stats['too_long_output']}")


def normalize_key(text):
    """Normalize text for dedup comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


seen_instructions = {}
seen_outputs = {}
deduped = []
dup_instruction = 0
dup_output = 0

MAX_SAME_OUTPUT = 3

for ex in cleaned_examples:
    inst_key = normalize_key(ex["instruction"])
    out_key = normalize_key(ex["output"])[:120]

    if inst_key in seen_instructions:
        dup_instruction += 1
        continue

    out_count = seen_outputs.get(out_key, 0)
    if out_count >= MAX_SAME_OUTPUT:
        dup_output += 1
        continue

    seen_instructions[inst_key] = ex["output"]
    seen_outputs[out_key] = out_count + 1
    deduped.append(ex)

print(f"  Before dedup: {len(cleaned_examples)}")
print(f"  Removed - duplicate instructions: {dup_instruction}")
print(f"  Removed - repeated outputs (>{MAX_SAME_OUTPUT}x): {dup_output}")
print(f"  After dedup:  {len(deduped)}")


import random
random.seed(42)
random.shuffle(deduped)

final = [{"instruction": ex["instruction"], "output": ex["output"]} for ex in deduped]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in final:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"  Saved: {OUTPUT_FILE}")
print(f"  Total examples: {len(final)}")

valid = 0
invalid = 0
with open(OUTPUT_FILE) as f:
    for line in f:
        try:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "output" in obj
            assert len(obj["instruction"]) >= MIN_INSTRUCTION_LEN
            assert len(obj["output"]) >= MIN_OUTPUT_LEN
            valid += 1
        except Exception:
            invalid += 1

source_counts = {}
for ex in deduped:
    src = os.path.basename(ex["_source"])
    source_counts[src] = source_counts.get(src, 0) + 1

print(f"  Valid:   {valid}")
print(f"  Invalid: {invalid}")
print(f"\n  Examples per source file:")
for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"    {src}: {count}")

print(f"\n{'='*55}")
print(f"DONE! Final file: {OUTPUT_FILE}")
print(f"Total clean examples: {valid}")
print(f"{'='*55}")
print("""
Next step: run clean_and_merge.py to combine with FuseChat
and create your final training dataset.
""")
