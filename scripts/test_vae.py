#!/usr/bin/env python3
"""测试VAE加载"""
import os, torch

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_CACHE_DIR = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models"

# 检查Dataset里有没有vae目录
print("检查Dataset里的模型目录:")
for d in os.listdir(MODEL_CACHE_DIR):
    dp = f"{MODEL_CACHE_DIR}/{d}"
    if os.path.isdir(dp):
        files = os.listdir(dp)
        print(f"  {d}: {files}")

# 尝试下载sd-vae-ft-mse
print("\n尝试加载VAE:")
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    print(f"  ✅ 从HF下载成功! {type(vae)}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# 检查from_single_file加载后的vae状态
print("\n检查from_single_file的vae:")
sd_file = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        sd_file, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    )
    print(f"  vae类型: {type(pipe.vae)}")
    print(f"  vae是None: {pipe.vae is None}")
except Exception as e:
    print(f"  ❌ {e}")
