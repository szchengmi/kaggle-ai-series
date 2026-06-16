#!/usr/bin/env python3
"""验证Dataset模型加载 — 允许diffusers下载config"""
import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import os
# 不完全离线 — 允许diffusers下载配置文件，但不下载模型权重
os.environ["HF_HUB_OFFLINE"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 搜索模型路径
MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        if os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5")):
            MODEL_CACHE_DIR = _candidate
            break

print(f"模型路径: {MODEL_CACHE_DIR}")

# 测试 SD 1.5 (from_single_file)
print("\n[1/3] 加载 SD 1.5 (from_single_file)...")
sd_file = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        sd_file, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.to(device)
    print(f"  ✅ SD 1.5 成功! ({device})")
    del pipe
except Exception as e:
    print(f"  ❌ {e}")

# 测试 AnimateDiff
print("\n[2/3] 加载 AnimateDiff...")
ad_path = f"{MODEL_CACHE_DIR}/animatediff"
try:
    from diffusers import MotionAdapter
    adapter = MotionAdapter.from_pretrained(ad_path, local_files_only=True, cache_dir=ad_path)
    print(f"  ✅ AnimateDiff 成功!")
    del adapter
except Exception as e:
    print(f"  ❌ {e}")

# 测试 Qwen tokenizer
print("\n[3/3] 加载 Qwen2.5-3B tokenizer...")
qwen_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
try:
    from transformers import AutoTokenizer
    # tokenizer用离线
    old = os.environ.get("TRANSFORMERS_OFFLINE", "")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    os.environ["TRANSFORMERS_OFFLINE"] = old
    print(f"  ✅ Qwen tokenizer 成功!")
except Exception as e:
    print(f"  ❌ {e}")

print("\n完成!")
