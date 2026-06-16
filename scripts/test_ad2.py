#!/usr/bin/env python3
"""诊断AnimateDiff motion_adapter兼容性"""
import os, torch, json

os.environ["HF_HUB_OFFLINE"] = "0"

MODEL_CACHE_DIR = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models"
ad_path = f"{MODEL_CACHE_DIR}/animatediff"

# 检查config
print("AnimateDiff config:")
with open(f"{ad_path}/config.json") as f:
    cfg = json.load(f)
for k, v in cfg.items():
    print(f"  {k}: {v}")

# 检查diffusers版本和MotionAdapter期望的config
print("\ndiffusers版本:")
import diffusers
print(f"  {diffusers.__version__}")

# 检查MotionAdapter期望哪些config字段
print("\nMotionAdapter期望的config字段:")
from diffusers import MotionAdapter
import inspect
sig = inspect.signature(MotionAdapter.__init__)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.default}")

# 加载motion_adapter后检查unet是否有motion模块
print("\n检查motion_adapter加载:")
adapter = MotionAdapter.from_pretrained(ad_path, torch_dtype=torch.float16, local_files_only=True, cache_dir=ad_path)
print(f"  adapter类型: {type(adapter)}")
print(f"  adapter属性: {[a for a in dir(adapter) if not a.startswith('_')]}")
