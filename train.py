"""Gemma 4 31B-it LoRA(QLoRA) 미세조정.

예: python train.py
    python train.py --data data/train.jsonl --epochs 3 --no-4bit
"""
import argparse
import json
import math
import os
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-4-31B-it")
p.add_argument("--data", default=os.path.join(BASE_DIR, "data/train.jsonl"))
p.add_argument("--output", default=os.path.join(BASE_DIR, "outputs/lora"))
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--grad-accum", type=int, default=4)
p.add_argument("--max-len", type=int, default=1024)
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--no-4bit", action="store_true", help="4비트 양자화 없이 bf16으로 학습 (GPU 메모리 80GB 이상 여러 장 필요)")
args = p.parse_args()

torch.manual_seed(0)
random.seed(0)

processor = AutoProcessor.from_pretrained(args.model)
tokenizer = processor.tokenizer

load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
if not args.no_4bit:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
model.config.use_cache = False

if not args.no_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

lora = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_r * 2,
    lora_dropout=0.05,
    # 비전 인코더는 제외하고 언어 모델 레이어에만 LoRA 적용
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()


def encode(messages):
    prompt = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][: args.max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return torch.tensor([full_ids]), torch.tensor([labels])


with open(args.data, encoding="utf-8") as f:
    samples = [encode(json.loads(line)["messages"]) for line in f if line.strip()]
print(f"학습 샘플 {len(samples)}개")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=args.lr)
total_steps = math.ceil(len(samples) / args.grad_accum) * args.epochs
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / total_steps))
device = next(model.parameters()).device

model.train()
step = 0
for epoch in range(args.epochs):
    random.shuffle(samples)
    running = 0.0
    for i, (input_ids, labels) in enumerate(samples):
        out = model(input_ids=input_ids.to(device), labels=labels.to(device))
        (out.loss / args.grad_accum).backward()
        running += out.loss.item()
        if (i + 1) % args.grad_accum == 0 or i + 1 == len(samples):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
    print(f"epoch {epoch + 1}/{args.epochs}  loss {running / len(samples):.4f}")

os.makedirs(args.output, exist_ok=True)
model.save_pretrained(args.output)
print(f"어댑터 저장: {args.output}")
