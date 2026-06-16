#!/usr/bin/env python3
"""精确检测 — CPU模式，先import torch"""
import torch  # 必须先import
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 搜索模型路径
print("\n路径搜索:")
MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        has_sd = os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5"))
        print(f"  找到: {_candidate} {'✅' if has_sd else '❌'}")
        if has_sd:
            MODEL_CACHE_DIR = _candidate
            break

print(f"MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")

# 测试SD 1.5
sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
print(f"\n[1/3] SD 1.5: {sd_path}")
print(f"  文件: {os.listdir(sd_path)}")
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=True, cache_dir=sd_path,
    )
    print("  ✅ 成功!"); del pipe
except Exception as e:
    print(f"  ❌ {e}")

# 测试AnimateDiff
ad_path = f"{MODEL_CACHE_DIR}/animatediff"
print(f"\n[2/3] AnimateDiff: {ad_path}")
print(f"  文件: {os.listdir(ad_path)}")
try:
    from diffusers import MotionAdapter
    adapter = MotionAdapter.from_pretrained(ad_path, local_files_only=True, cache_dir=ad_path)
    print("  ✅ 成功!"); del adapter
except Exception as e:
    print(f"  ❌ {e}")

# 测试Qwen tokenizer
qwen_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
print(f"\n[3/3] Qwen: {qwen_path}")
print(f"  文件: {os.listdir(qwen_path)}")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    print("  ✅ 成功!")
except Exception as e:
    print(f"  ❌ {e}")

print("\n完成")
