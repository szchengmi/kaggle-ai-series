#!/usr/bin/env python3
"""精确检测 — 模拟pipeline的路径搜索逻辑"""
import os

print("模拟 pipeline 路径搜索:")
print("=" * 60)

MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"  # fallback
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        has_sd = os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5"))
        print(f"  找到: {_candidate}")
        print(f"    stable-diffusion-v1-5: {'✅' if has_sd else '❌'}")
        if has_sd:
            MODEL_CACHE_DIR = _candidate
            print(f"  → 选中: {MODEL_CACHE_DIR}")
            break

print(f"\n最终 MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")

# 测试 from_pretrained 能否加载
print("\n测试 from_pretrained 加载:")
print("-" * 60)

sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
print(f"SD路径: {sd_path}")
print(f"  存在: {os.path.isdir(sd_path)}")
print(f"  文件: {[f for f in os.listdir(sd_path) if os.path.isfile(f'{sd_path}/{f}')]}")

# 测试diffusers能否识别
try:
    from diffusers import StableDiffusionPipeline
    print("\n尝试加载 SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        torch_dtype=__import__('torch').float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        cache_dir=sd_path,
    )
    print("✅ SD 1.5 加载成功!")
    del pipe
    import torch; torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ SD 1.5 加载失败: {e}")

# 测试Qwen
qwen_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
print(f"\nQwen路径: {qwen_path}")
print(f"  存在: {os.path.isdir(qwen_path)}")
if os.path.isdir(qwen_path):
    print(f"  文件: {[f for f in os.listdir(qwen_path) if os.path.isfile(f'{qwen_path}/{f}')]}")

try:
    from transformers import AutoTokenizer
    print("\n尝试加载 Qwen tokenizer...")
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    print("✅ Qwen tokenizer 加载成功!")
except Exception as e:
    print(f"❌ Qwen tokenizer 加载失败: {e}")

print("\n" + "=" * 60)
