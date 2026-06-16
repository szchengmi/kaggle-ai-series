#!/usr/bin/env python3
"""检查Qwen目录文件"""
import os

qwen_path = "/kaggle/input/datasets/saysnkaggle/newdataset/kaggle-ai-series/models/Qwen2.5-3B-Instruct"
print(f"目录: {qwen_path}")
print(f"存在: {os.path.isdir(qwen_path)}")

if os.path.isdir(qwen_path):
    for f in sorted(os.listdir(qwen_path)):
        fp = f"{qwen_path}/{f}"
        size = os.path.getsize(fp) if os.path.isfile(fp) else 0
        print(f"  {f} ({size/1e6:.1f}MB)")
