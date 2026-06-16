#!/usr/bin/env python3
"""精确检测 — 先升级torch再测试模型加载"""
import os
import subprocess
import sys

# Step 1: 升级torch（解决MKL冲突）
print("升级 PyTorch...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     "torch", "torchvision", "torchaudio",
     "--index-url", "https://download.pytorch.org/whl/cu124"],
    timeout=300
)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Step 2: 搜索模型路径
print("\n模拟 pipeline 路径搜索:")
print("=" * 60)

MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        has_sd = os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5"))
        print(f"  找到: {_candidate}")
        if has_sd:
            MODEL_CACHE_DIR = _candidate
            print(f"  → 选中!")
            break

print(f"MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")

# Step 3: 测试加载
print("\n测试 from_pretrained:")
print("-" * 60)

# SD 1.5
sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
print(f"\n[1/3] SD 1.5: {sd_path}")
print(f"  文件: {os.listdir(sd_path)}")
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=True, cache_dir=sd_path,
    )
    print("  ✅ 成功!"); del pipe; torch.cuda.empty_cache()
except Exception as e:
    print(f"  ❌ {e}")

# AnimateDiff
ad_path = f"{MODEL_CACHE_DIR}/animatediff"
print(f"\n[2/3] AnimateDiff: {ad_path}")
print(f"  文件: {os.listdir(ad_path)}")
try:
    from diffusers import MotionAdapter
    adapter = MotionAdapter.from_pretrained(ad_path, torch_dtype=torch.float16, local_files_only=True, cache_dir=ad_path)
    print("  ✅ 成功!"); del adapter
except Exception as e:
    print(f"  ❌ {e}")

# Qwen
qwen_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
print(f"\n[3/3] Qwen2.5-3B: {qwen_path}")
print(f"  文件: {[f for f in os.listdir(qwen_path)]}")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    print("  ✅ 成功!")
except Exception as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 60)
