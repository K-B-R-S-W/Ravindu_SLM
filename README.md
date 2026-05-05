# Ravindu_SLM

A personal small language model (SLM) fine-tuned on **Qwen 2.5 1.5B Instruct** using LoRA via Unsloth. Built to act as a personal AI assistant with knowledge of my projects, skills, career background, and Sri Lankan context.

---

## Demo

```
You: what is CrimeGuard?
AI:  CrimeGuard is my final year dissertation — a multi-layered AI public safety system.
     Combines ArcFace face recognition (95% accuracy), YOLOv11 for weapon detection,
     a multilingual emergency chatbot in English/Sinhala/Tamil using LangGraph,
     and an ESP32 IoT alert system. FastAPI WebSocket backend keeps everything
     under 500ms latency.

You: how do i load a csv in pandas?
AI:  import pandas as pd
     df = pd.read_csv('file.csv')
     print(df.head())
```

---

## Model Details

| | |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning method | LoRA (Low-Rank Adaptation) |
| Training framework | Unsloth + TRL SFTTrainer |
| Hardware | Google Colab T4 (15GB VRAM) |
| Training time | 52 minutes |

---

## Training Configuration

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target layers | q, k, v, o, gate, up, down projections |
| Quantization | 4-bit (QLoRA) |
| Epochs | 3 |
| Effective batch size | 8 (2 × 4 gradient accumulation) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine with warmup |
| Optimizer | Paged AdamW 32-bit |
| Max sequence length | 2048 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Trainable parameters | 18.46M / 1,562M (1.18%) |

---

## Dataset

| Source | Examples | Coverage |
|---|---|---|
| Personal data (custom) | 803 | Projects, bio, Sri Lanka context, career |
| FuseChat-Mixture (HuggingFace) | 2,000 | General chat, coding, reasoning |
| **Total** | **2,803** | |

Personal data was generated and curated manually, then cleaned and deduplicated using `merge_personal.py`. FuseChat samples were drawn evenly across the full 95k dataset to ensure category balance.

---

## Training Results

| Metric | Value |
|---|---|
| Final train loss | 0.9000 |
| Best eval loss | 1.0313 |
| Final perplexity | **2.80** |
| Loss gap (train vs val) | 0.14 — healthy generalization |
| Early stopping | Triggered at epoch 3 (no overfitting) |

Perplexity of **2.80** on the validation set indicates the model learned to predict responses confidently without memorizing the training data.

---

## Evaluation

Tested across 19 questions covering personal knowledge, Python reasoning, math, and logic.

| Category | Score |
|---|---|
| Python / code reasoning | Strong |
| Basic arithmetic | Strong |
| String operations | Strong |
| Logic / transitive reasoning | Good |
| Personal facts (projects, dates) | Moderate |
| Multi-step computation | Needs improvement |
| **Overall** | **61 / 100** |

---

## Repo Structure

```
Ravindu_SLM/
├── SLM_Train.ipynb              # Main training notebook (22 cells)
├── ravindu_training_notebook.py # Training script version
├── merge_personal.py            # Merge + clean personal JSONL files
├── clean_and_merge.py           # Combine personal data + FuseChat
└── README.md
```

---

## How to Use

### Load the model in Colab

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/ravindu_ai_final_merged",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### Chat

```python
SYSTEM = "You are Ravindu Sankalpa's personal AI assistant."

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": "tell me about your projects"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```
