"""Run Qwen inference and score candidate labels."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.io import get_logger

LOG = get_logger(__name__)

LABEL_CANDIDATES = ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO")


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        name.lower()]


def available_vram_gb() -> Optional[float]:
    """Free VRAM right now. Used to fail early instead of dying mid-run."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


class Reader:
    def __init__(self, model_name: str, quant: Optional[Dict[str, Any]] = None,
                 adapter_path: Optional[str] = None, device: str = "cuda",
                 dtype: str = "bfloat16", revision: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torch and transformers are required for the reader.\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                "  pip install transformers peft bitsandbytes accelerate"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"device_map": {"": device}, "low_cpu_mem_usage": True}
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Run scripts/00_check_env.py before inference.")
        if quant and quant.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            compute_dtype = _torch_dtype(quant.get("compute_dtype", dtype))
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
                bnb_4bit_compute_dtype=compute_dtype,
            )
            kwargs["torch_dtype"] = compute_dtype
        else:
            kwargs["torch_dtype"] = _torch_dtype(dtype)

        free = available_vram_gb()
        if free is not None and free < 2.5:
            LOG.warning("only %.2f GB VRAM free; 4-bit 1.5B needs about 3.4 GB peak", free)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
        self.model.eval()

        if adapter_path:
            from peft import PeftModel

            LOG.info("attaching adapter %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        self.model_name = model_name
        self.adapter_path = adapter_path

    def chat_prompt(self, prompt: str) -> str:
        return chat_prompt(self.tokenizer, prompt)

    # -- generation ------------------------------------------------------------------
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 160,
                 batch_size: int = 4, use_adapter: bool = True) -> List[str]:
        outputs: List[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = [self.chat_prompt(p) for p in prompts[start:start + batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            adapter_context = (self.model.disable_adapter()
                               if self.adapter_path and not use_adapter else nullcontext())
            with adapter_context, self.torch.inference_mode():
                ids = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen = ids[:, enc["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outputs

    # -- teacher-forced label probabilities -------------------------------------------
    def score_labels(self, prompt: str, candidates: Sequence[str] = LABEL_CANDIDATES,
                     assistant_prefix: str = "") -> Dict[str, float]:
        """Compare complete label strings as alternative continuations."""
        text = self.chat_prompt(prompt) + assistant_prefix
        prompt_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        prompt_ids = prompt_ids.to(self.model.device)
        logps: List[float] = []
        with self.torch.inference_mode():
            prefix = self.model(input_ids=prompt_ids, logits_to_keep=1, use_cache=True)
            cache = prefix.past_key_values
            for cand in candidates:
                suffix = (" " if assistant_prefix else "") + cand
                ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                cand_ids = self.torch.tensor([ids], device=self.model.device)
                logits = prefix.logits
                if len(ids) > 1:
                    tail = self.model(input_ids=cand_ids[:, :-1], past_key_values=cache,
                                      use_cache=True).logits
                    logits = self.torch.cat([logits, tail], dim=1)
                    cache.crop(prompt_ids.shape[1])
                log_probs = self.torch.log_softmax(logits.float(), dim=-1)
                logps.append(float(log_probs.gather(-1, cand_ids.unsqueeze(-1)).sum()))

        arr = np.asarray(logps, dtype=np.float64)
        arr = arr - arr.max()
        probs = np.exp(arr) / np.exp(arr).sum()
        return {cand: float(p) for cand, p in zip(candidates, probs)}


def chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def labels_to_predictions(scores: Dict[str, float]) -> Dict[str, Any]:
    """Pick the argmax label and report the probability of that label (the gate's input)."""
    best = max(scores, key=scores.get)
    mapped = {"NOT_ENOUGH_INFO": "NEI"}.get(best, best)
    return {
        "pred_label": mapped,
        "pred_prob": float(scores[best]),
        "probs": {("NEI" if k == "NOT_ENOUGH_INFO" else k): float(v) for k, v in scores.items()},
    }


def free_vram_report() -> Dict[str, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": 0.0}
        free, total = torch.cuda.mem_get_info()
        return {"free_gb": free / 1024 ** 3, "total_gb": total / 1024 ** 3,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
    except Exception:
        return {"available": 0.0}
