#!/usr/bin/env python3
"""精确检测 — 验证Dataset模型加载 (CPU/GPU自适应)"""
import torch  # 必须先import
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"设备: {device} | 精度: {DTYPE}")

# 搜索模型路径
print("\n" + "=" * 60)
print("路径搜索")
print("=" * 60)

MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        has_sd = os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5"))
        print(f"  找到: {_candidate}")
        if has_sd:
            MODEL_CACHE_DIR = _candidate
            break

print(f"✅ MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")

# 检测模型文件
print("\n" + "=" * 60)
print("模型文件检测")
print("=" * 60)

models = {
    "SD 1.5": f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5",
    "AnimateDiff": f"{MODEL_CACHE_DIR}/animatediff",
    "Qwen2.5-3B": f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct",
}

for name, path in models.items():
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]
        size = sum(os.path.getsize(f"{path}/{f}") for f in files) / 1e6
        print(f"  ✅ {name}: {path} ({size:.0f}MB, {len(files)}个文件)")
        for ff in files:
            print(f"     - {ff}")
    else:
        print(f"  ❌ {name}: {path} 不存在")

# 测试 from_pretrained
print("\n" + "=" * 60)
print("from_pretrained 加载测试")
print("=" * 60)

# SD 1.5
sd_path = f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5"
print(f"\n[1/3] SD 1.5: {sd_path}")
try:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path, torch_dtype=DTYPE,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=True, cache_dir=sd_path,
    )
    pipe.to(device)
    print(f"  ✅ SD 1.5 加载成功! ({device})")
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except Exception as e:
    print(f"  ❌ 失败: {e}")

# AnimateDiff
ad_path = f"{MODEL_CACHE_DIR}/animatediff"
print(f"\n[2/3] AnimateDiff: {ad_path}")
try:
    from diffusers import MotionAdapter
    adapter = MotionAdapter.from_pretrained(ad_path, torch_dtype=DTYPE, local_files_only=True, cache_dir=ad_path)
    print(f"  ✅ AnimateDiff 加载成功!")
    del adapter
except Exception as e:
    print(f"  ❌ 失败: {e}")

# Qwen2.5-3B (只测tokenizer，模型太大)
qwen_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
print(f"\n[3/3] Qwen2.5-3B: {qwen_path}")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)
    print(f"  ✅ Qwen tokenizer 加载成功!")
except Exception as e:
    print(f"  ❌ 失败: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
