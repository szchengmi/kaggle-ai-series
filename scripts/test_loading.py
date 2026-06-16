#!/usr/bin/env python3
"""验证Dataset模型加载 (CPU模式)"""
import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

# 检测模型文件
for name in ["SD 1.5", "AnimateDiff", "Qwen2.5-3B"]:
    paths = {
        "SD 1.5": f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5",
        "AnimateDiff": f"{MODEL_CACHE_DIR}/animatediff",
        "Qwen2.5-3B": f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct",
    }
    p = paths[name]
    if os.path.isdir(p):
        files = os.listdir(p)
        size = sum(os.path.getsize(f"{p}/{f}") for f in files if os.path.isfile(f"{p}/{f}")) / 1e6
        print(f"  ✅ {name}: {size:.0f}MB, {len(files)}个文件")
    else:
        print(f"  ❌ {name}: 不存在")

# 测试 SD 1.5 (from_single_file)
print("\n[1/3] 加载 SD 1.5 (from_single_file)...")
sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
sd_file = f"{sd_path}/v1-5-pruned-emaonly.safetensors"
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
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    print(f"  ✅ Qwen tokenizer 成功!")
except Exception as e:
    print(f"  ❌ {e}")

print("\n完成!")
