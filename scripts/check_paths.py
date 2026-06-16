#!/usr/bin/env python3
"""诊断脚本 — 检查Dataset挂载路径和模型文件"""
import os

print("=" * 60)
print("路径诊断")
print("=" * 60)

# 1. 列出 /kaggle/input/ 完整结构
print("\n📁 /kaggle/input/ 完整结构:")
if os.path.isdir("/kaggle/input"):
    for d in sorted(os.listdir("/kaggle/input")):
        full = f"/kaggle/input/{d}"
        if not os.path.isdir(full):
            continue
        print(f"  📂 {d}/")
        for sub in sorted(os.listdir(full)):
            sub_path = f"{full}/{sub}"
            if os.path.isdir(sub_path):
                print(f"    📂 {sub}/")
                for item in sorted(os.listdir(sub_path)):
                    item_path = f"{sub_path}/{item}"
                    tag = "📂" if os.path.isdir(item_path) else "📄"
                    extra = ""
                    if os.path.isdir(item_path) and "models" in item.lower():
                        extra = " ← models目录!"
                    print(f"      {tag} {item}{extra}")
            else:
                print(f"    📄 {sub}")
else:
    print("  ❌ /kaggle/input/ 不存在")

# 2. 搜索所有 models/ 目录
print("\n🔍 搜索所有 models/ 目录:")
found_models = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "models" in dirs:
        models_path = f"{root}/models"
        print(f"  ✅ {models_path}")
        for m in sorted(os.listdir(models_path)):
            mp = f"{models_path}/{m}"
            if os.path.isdir(mp):
                size = sum(os.path.getsize(f"{mp}/{f}") for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")) / 1e6
                print(f"    📂 {m}/ ({size:.0f}MB)")
        found_models.append(models_path)

if not found_models:
    print("  ❌ 没有找到任何 models/ 目录")

print("\n" + "=" * 60)
