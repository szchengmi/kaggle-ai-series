#!/usr/bin/env python3
"""诊断脚本 — 检查Dataset挂载路径和模型文件"""
import os

print("=" * 60)
print("路径诊断")
print("=" * 60)

# 1. 检查 /kaggle/input/ 下有什么
print("\n📁 /kaggle/input/ 内容:")
if os.path.isdir("/kaggle/input"):
    for d in sorted(os.listdir("/kaggle/input")):
        full = f"/kaggle/input/{d}"
        print(f"  📂 {d}/")
        # 检查里面有没有 kaggle-ai-series/models
        models_path = f"{full}/kaggle-ai-series/models"
        if os.path.isdir(models_path):
            print(f"    ✅ kaggle-ai-series/models/ 存在!")
            for m in sorted(os.listdir(models_path)):
                mp = f"{models_path}/{m}"
                if os.path.isdir(mp):
                    size = sum(os.path.getsize(f"{mp}/{f}") for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")) / 1e6
                    files = [f for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")]
                    print(f"      📂 {m}/ ({size:.0f}MB, {len(files)}个文件)")
                    for ff in files:
                        print(f"         - {ff}")
        else:
            # 检查直接子目录里有没有models
            for sub in sorted(os.listdir(full)):
                sub_models = f"{full}/{sub}/models"
                if os.path.isdir(sub_models):
                    print(f"    ✅ {sub}/models/ 存在!")
                    for m in sorted(os.listdir(sub_models)):
                        print(f"      📂 {m}/")
else:
    print("  ❌ /kaggle/input/ 不存在")

# 2. 检查 /kaggle/working/ 下有什么
print("\n📁 /kaggle/working/ 内容:")
if os.path.isdir("/kaggle/working"):
    for d in sorted(os.listdir("/kaggle/working")):
        print(f"  {'📂' if os.path.isdir(f'/kaggle/working/{d}') else '📄'} {d}")

# 3. 检查模型文件
print("\n🔍 模型文件检查:")
base = "/kaggle/input"
for ds in sorted(os.listdir(base)) if os.path.isdir(base) else []:
    models_dir = f"{base}/{ds}/kaggle-ai-series/models"
    if os.path.isdir(models_dir):
        for model_name in ["stable-diffusion-v1-5", "animatediff", "Qwen2.5-3B-Instruct"]:
            mp = f"{models_dir}/{model_name}"
            if os.path.isdir(mp):
                files = [f for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")]
                print(f"  ✅ {ds}/kaggle-ai-series/models/{model_name}/ ({len(files)}个文件)")
            else:
                print(f"  ❌ {ds}/kaggle-ai-series/models/{model_name}/ 不存在")

print("\n" + "=" * 60)
