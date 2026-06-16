#!/usr/bin/env python3
"""检查AnimateDiff config和SD输出"""
import os, torch, json

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_CACHE_DIR = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models"

# 检查AnimateDiff config
ad_path = f"{MODEL_CACHE_DIR}/animatediff"
print("AnimateDiff config:")
with open(f"{ad_path}/config.json") as f:
    cfg = json.load(f)
    print(f"  motion_num_hidden_layers: {cfg.get('motion_num_hidden_layers')}")
    print(f"  motion_mid_block_layers: {cfg.get('motion_mid_block_layers')}")
    print(f"  unet_use_cross_frame_attention: {cfg.get('unet_use_cross_frame_attention')}")

# 快速测试SD生成一张图
print("\n测试SD生成:")
sd_file = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_single_file(
    sd_file, torch_dtype=torch.float16,
    safety_checker=None, requires_safety_checker=False,
)
pipe.to("cuda")
pipe.enable_attention_slicing()

result = pipe(
    prompt="a beautiful anime girl standing in a garden, masterpiece, best quality",
    negative_prompt="blurry, low quality",
    width=512, height=512,
    num_inference_steps=20,
    guidance_scale=7.5,
)
print(f"  结果类型: {type(result)}")
print(f"  images: {result.images}")
if result.images:
    result.images[0].save("/kaggle/working/test_sd_output.png")
    print(f"  ✅ 图像已保存到 /kaggle/working/test_sd_output.png")
    import os
    size = os.path.getsize("/kaggle/working/test_sd_output.png")
    print(f"  文件大小: {size/1e3:.1f}KB")
else:
    print(f"  ❌ 无图像输出")

# 检查图像是否全黑
if result.images:
    import numpy as np
    from PIL import Image
    img = np.array(result.images[0])
    print(f"  像素均值: {img.mean():.1f} (0=全黑, 255=全白)")
    print(f"  像素范围: {img.min()} - {img.max()}")
    if img.mean() < 5:
        print(f"  ⚠️ 图像几乎全黑!")
