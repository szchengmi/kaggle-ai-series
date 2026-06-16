#!/usr/bin/env python3
"""
模型打包脚本
把 /kaggle/working/output/models/ 复制到 /kaggle/working/kaggle-ai-series/models/
这样Save as Dataset时模型就包含在内了
"""

import os
import shutil

SRC = "/kaggle/working/output/models"
DST = "/kaggle/working/kaggle-ai-series/models"

if not os.path.exists(SRC):
    print("❌ 源目录不存在")
    exit(1)

if os.path.exists(DST):
    print("清理旧的models...")
    shutil.rmtree(DST)

print(f"复制模型到代码仓库...")
shutil.copytree(SRC, DST)

# 验证
total = 0
for d in os.listdir(DST):
    full = f"{DST}/{d}"
    if os.path.isdir(full):
        files = [f for f in os.listdir(full) if os.path.isfile(f"{full}/{f}")]
        size = sum(os.path.getsize(f"{full}/{f}") for f in files)
        total += size
        print(f"  ✅ {d}/ ({len(files)} files, {size/1e9:.2f}GB)")

print(f"\n总计: {total/1e9:.2f}GB")
print(f"模型已复制到: {DST}")
print("现在去Kaggle Output面板 → Save as Dataset")
