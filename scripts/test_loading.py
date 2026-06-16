#!/usr/bin/env python3
"""最简诊断 — 只import torch看能不能活"""
print("开始...")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 搜索模型
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        has_sd = os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5"))
        if has_sd:
            print(f"模型路径: {_candidate}")
            for m in os.listdir(_candidate):
                mp = f"{_candidate}/{m}"
                if os.path.isdir(mp):
                    size = sum(os.path.getsize(f"{mp}/{f}") for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")) / 1e6
                    print(f"  {m}: {size:.0f}MB")
            break
print("完成!")
