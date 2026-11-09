"""Prepare supervised examples and train a seed-specific LoRA adapter."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .infer import chat_prompt
from .prompts import build_prompt
from ..utils.io import load_jsonl_as_list


def build_sft_examples(train_claims, sentences_by_claim):
    examples = []
    for claim in train_claims:
        label = claim["label"]
        if label is None:
            raise ValueError("Training requires labeled claims")
        sentences = sentences_by_claim.get(claim["claim_id"], [])
        examples.append({
            "claim_id": claim["claim_id"],
            "prompt": build_prompt("grounded_label_only", claim["claim"], sentences=sentences),
            "target": "NOT_ENOUGH_INFO" if label == "NEI" else label,
            "label": label,
            "has_evidence": bool(sentences),
        })
    return examples


def encode_example(tokenizer, example, max_length):
    prompt = tokenizer.encode(chat_prompt(tokenizer, example["prompt"]), add_special_tokens=False)
    target = tokenizer.encode(example["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > max_length:
        budget = max_length - len(target)
        head = budget // 4
        prompt = prompt[:head] + prompt[-(budget - head):]
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def train(config, train_file: Path, output_dir: Path, seed: int, smoke_test=False, resume=False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA-enabled PyTorch installation")
    reader = config["reader"]
    train_cfg, lora, quant = reader["train"], reader["lora"], reader["quant"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"device_map": {"": 0}, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if quant["load_in_4bit"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant["quant_type"],
            bnb_4bit_use_double_quant=quant["double_quant"], bnb_4bit_compute_dtype=dtype,
        )

    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(reader["model"], revision=reader.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(reader["model"], revision=reader.get("revision"), **kwargs)
    model.config.use_cache = False
    if quant["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))

    examples = load_jsonl_as_list(train_file)
    if smoke_test:
        examples = random.Random(seed).sample(examples, min(32, len(examples)))
    dataset = [encode_example(tokenizer, ex, int(train_cfg["max_seq_len"])) for ex in examples]

    def collate(batch):
        size = max(len(ex["input_ids"]) for ex in batch)
        ids, labels, masks = [], [], []
        for ex in batch:
            padding = size - len(ex["input_ids"])
            ids.append(ex["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(ex["labels"] + [-100] * padding)
            masks.append([1] * len(ex["input_ids"]) + [0] * padding)
        return {key: torch.tensor(val) for key, val in
                (("input_ids", ids), ("labels", labels), ("attention_mask", masks))}

    class CompletionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            start = int((labels != -100).nonzero()[:, 1].min()) - 1
            keep = labels.shape[1] - start
            outputs = model(**inputs, logits_to_keep=keep)
            logits = outputs.logits[:, :-1, :].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels[:, start + 1:].reshape(-1)
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(output_dir), max_steps=8 if smoke_test else -1,
        num_train_epochs=float(train_cfg["epochs"]), learning_rate=float(train_cfg["lr"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=2 if smoke_test else int(train_cfg["grad_accum"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]), max_grad_norm=float(train_cfg["grad_clip"]),
        lr_scheduler_type="cosine", save_strategy="no" if smoke_test else "epoch",
        save_total_limit=1, logging_steps=1 if smoke_test else 20,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", seed=seed, data_seed=seed, dataloader_num_workers=0,
        report_to=[], logging_nan_inf_filter=False, label_names=["labels"], disable_tqdm=True,
    )
    trainer = CompletionTrainer(model=model, args=args, train_dataset=dataset, data_collator=collate)
    trainer.model_accepts_loss_kwargs = False
    metrics = trainer.train(resume_from_checkpoint=resume or None).metrics
    loss = float(metrics["train_loss"])
    if not math.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: {loss}")
    trainer.save_model(str(output_dir))
    (output_dir / "train_meta.json").write_text(json.dumps({
        "seed": seed, "smoke_test": smoke_test, "n_examples": len(dataset),
        "lora": lora, "load_in_4bit": quant["load_in_4bit"], "metrics": metrics,
        "training_config": train_cfg,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "loss_trace": [row for row in trainer.state.log_history if "loss" in row],
    }, indent=2), encoding="utf-8")
    return output_dir
