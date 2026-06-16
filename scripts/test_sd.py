#!/usr/bin/env python3
"""测试SD 1.5 + VAE加载"""
import torch
import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_CACHE_DIR = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models"
sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
sd_file = f"{sd_path}/v1-5-pruned-emaonly.safetensors"

print(f"SD文件: {sd_file}")
print(f"存在: {os.path.isfile(sd_file)}")

# 方法1: from_single_file
print("\n[方法1] from_single_file:")
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        sd_file, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    )
    print(f"  pipe类型: {type(pipe)}")
    print(f"  vae: {type(pipe.vae)}")
    print(f"  unet: {type(pipe.unet)}")
    print(f"  text_encoder: {type(pipe.text_encoder)}")
    # 检查哪些组件是None
    for name in ['vae', 'unet', 'text_encoder', 'tokenizer', 'scheduler']:
        comp = getattr(pipe, name, None)
        status = "✅" if comp is not None else "❌ None"
        print(f"    {name}: {status}")
except Exception as e:
    print(f"  ❌ {e}")

# 方法2: 手动从safetensors加载所有组件
print("\n[方法2] 手动加载:")
try:
    from safetensors.torch import load_file
    from diffusers import StableDiffusionPipeline, AutoencoderCLIPTextModel, CLIPTokenizer, UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel
    
    # 加载safetensors
    print("  加载safetensors...")
    state_dict = load_file(sd_file, device="cpu")
    print(f"  权重keys: {len(state_dict)}")
    
    # 检查vae权重是否存在
    vae_keys = [k for k in state_dict if k.startswith("vae")]
    unet_keys = [k for k in state_dict if k.startswith("unet")]
    text_keys = [k for k in state_dict if k.startswith("text_encoder")]
    print(f"  vae keys: {len(vae_keys)}")
    print(f"  unet keys: {len(unet_keys)}")
    print(f"  text_encoder keys: {len(text_keys)}")
    
except Exception as e:
    print(f"  ❌ {e}")

print("\n完成!")
