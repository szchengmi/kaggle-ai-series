#!/usr/bin/env python3
"""测试AnimateDiff视频生成"""
import os, torch

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_CACHE_DIR = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models"
sd_file = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
ad_path = f"{MODEL_CACHE_DIR}/animatediff"

print("加载AnimateDiff...")
from diffusers import StableDiffusionPipeline, MotionAdapter, EulerAncestralDiscreteScheduler

adapter = MotionAdapter.from_pretrained(ad_path, torch_dtype=torch.float16, local_files_only=True, cache_dir=ad_path)
pipe = StableDiffusionPipeline.from_single_file(
    sd_file, motion_adapter=adapter, torch_dtype=torch.float16,
    safety_checker=None, requires_safety_checker=False,
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear", steps_offset=1)

print("生成视频...")
result = pipe(
    prompt="a beautiful anime girl standing in a garden, masterpiece, best quality, smooth animation",
    negative_prompt="blurry, low quality, static",
    width=512, height=512,
    num_frames=8,
    num_inference_steps=15,
    guidance_scale=7.5,
)

print(f"结果类型: {type(result)}")
print(f"有frames: {hasattr(result, 'frames')}")
print(f"有images: {hasattr(result, 'images')}")

if hasattr(result, 'frames'):
    frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames
    print(f"帧数: {len(frames)}")
    if frames:
        frames[0].save("/kaggle/working/test_ad_frame0.png")
        import numpy as np
        from PIL import Image
        img = np.array(frames[0])
        print(f"像素均值: {img.mean():.1f}")
        print(f"像素范围: {img.min()} - {img.max()}")
        if img.mean() < 5:
            print("⚠️ 帧全黑!")
        else:
            print("✅ 帧有内容!")
elif hasattr(result, 'images'):
    print(f"图像数: {len(result.images)}")
    result.images[0].save("/kaggle/working/test_ad_frame0.png")

print("完成!")
