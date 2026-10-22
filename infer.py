"""Gemma 4 31B-it 추론.

예: python infer.py --prompt "LoRA가 뭐야?"
    python infer.py --prompt "LoRA가 뭐야?" --adapter outputs/lora
"""
import argparse
import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-4-31B-it")
p.add_argument("--prompt", default="LoRA가 뭐야?")
p.add_argument("--adapter", default=None, help="train.py로 만든 LoRA 어댑터 경로")
p.add_argument("--max-new-tokens", type=int, default=512)
p.add_argument("--4bit", dest="four_bit", action="store_true", help="4비트로 올려 GPU 메모리 사용량 줄이기")
args = p.parse_args()

processor = AutoProcessor.from_pretrained(args.model)

load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
if args.four_bit:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)

if args.adapter:
    from peft import PeftModel

    adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(BASE_DIR, args.adapter)
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

messages = [{"role": "user", "content": args.prompt}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

print(processor.decode(outputs[0][input_len:], skip_special_tokens=True))
