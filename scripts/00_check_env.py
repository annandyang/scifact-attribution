"""Check only the packages needed for the requested stage."""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "gpu", "demo"), default="core")
    args = parser.parse_args()

    packages = {
        "core": ["numpy", "scipy", "yaml"],
        "gpu": ["numpy", "scipy", "yaml", "torch", "transformers", "accelerate",
                "bitsandbytes", "peft", "sentence_transformers"],
        "demo": ["yaml", "gradio"],
    }
    print(f"Python {sys.version.split()[0]}: {sys.executable}")
    ok = sys.version_info[:2] == (3, 11)
    if not ok:
        print("This Windows setup expects Python 3.11.")
    for name in packages[args.stage]:
        try:
            module = importlib.import_module(name)
            print(f"  OK  {name}: {getattr(module, '__version__', '')}")
        except Exception as error:
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
            ok = False

    if args.stage == "gpu" and ok:
        import bitsandbytes as bnb
        import torch

        if not torch.cuda.is_available():
            print("FAIL: CUDA is not available. Check the NVIDIA driver and CUDA PyTorch wheel.")
            ok = False
        else:
            free, total = torch.cuda.mem_get_info()
            print(f"{torch.cuda.get_device_name()}: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free")
            print(f"CUDA runtime {torch.version.cuda}; BF16: {torch.cuda.is_bf16_supported()}")
            try:
                layer = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.float16)
                result = layer.to("cuda")(torch.zeros(1, 64, device="cuda", dtype=torch.float16))
                torch.cuda.synchronize()
                ok = bool(torch.isfinite(result).all())
                print("NF4 CUDA kernel:", "OK" if ok else "FAIL")
            except Exception as error:
                print(f"FAIL: NF4 CUDA kernel: {error}")
                ok = False
            if free < 4 * 2**30:
                print("Close GPU-heavy apps before inference or training; less than 4 GiB is free.")
    disk = shutil.disk_usage(Path(__file__).resolve().parents[1])
    print(f"Disk free: {disk.free / 2**30:.1f} GiB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
