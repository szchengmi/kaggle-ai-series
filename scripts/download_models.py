#!/usr/bin/env python3
"""
Kaggle 模型下载脚本
====================
一次性下载所有需要的模型到 /kaggle/working/output/models/
下载完成后保存到Kaggle Dataset，后续直接挂载使用。

在Kaggle Notebook中运行：
    !python download_models.py
"""

import os
import sys
import time
import subprocess

# ============================================================
# 配置
# ============================================================

MODEL_CACHE_DIR = "/kaggle/working/output/models"
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

# HuggingFace Token（从Kaggle Secrets读取）
def get_kaggle_secret(key_name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass
    # Kaggle自动注入的变量
    try:
        if key_name == "HF_TOKEN":
            return hf_token  # noqa: F821
    except:
        pass
    return ""

HF_TOKEN = get_kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("[OK] HF_TOKEN 已设置")
else:
    print("[WARN] HF_TOKEN 未设置， gated model可能无法下载")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def download_model(model_id, dir_name=None):
    """用huggingface-cli下载模型到指定目录"""
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    
    if os.path.exists(target) and len(os.listdir(target)) > 0:
        log(f"[SKIP] {model_id} 已存在")
        return
    
    log(f"[DOWNLOAD] {model_id} → {target}")
    os.makedirs(target, exist_ok=True)
    
    # 用huggingface-cli下载（比Python API更稳定）
    cmd = f'huggingface-cli download {model_id} --local-dir {target} --resume-download'
    r = run_cmd(cmd, timeout=600)
    
    if r.returncode == 0 and os.path.exists(target) and len(os.listdir(target)) > 0:
        size = sum(os.path.getsize(f"{target}/{f}") for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")) / 1e9
        log(f"[OK] {model_id} ({size:.2f}GB)")
    else:
        log(f"[FAIL] {model_id}: {r.stderr[:200] if r.stderr else 'unknown error'}")


# ============================================================
# 模型列表
# ============================================================

MODELS = [
    # ---- Step 3: 画面生成 ----
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "desc": "Stable Diffusion 1.5 (2.43GB)"
    },
    {
        "id": "stabilityai/sd-vae-ft-mse",
        "name": "VAE ft-mse",
        "desc": "VAE改进版 (替代默认VAE)"
    },
    
    # ---- Step 4: 视频生成 ----
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff Motion",
        "desc": "AnimateDiff Motion Adapter (301MB)"
    },
    
    # ---- Step 1备选: 本地LLM ----
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "desc": "Qwen2.5-3B-Instruct (6.44GB, Gemini被封时备用)"
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载")
    log("=" * 55)
    log(f"目标目录: {MODEL_CACHE_DIR}")
    log(f"模型数量: {len(MODELS)}")
    log("=" * 55)
    
    # 安装huggingface-cli
    log("检查 huggingface-cli...")
    r = run_cmd("huggingface-cli --version")
    if r.returncode != 0:
        log("安装 huggingface_hub[cli]...")
        run_cmd("pip install -q -U 'huggingface_hub[hf_transfer]'", timeout=120)
        run_cmd("pip install -q hf_transfer", timeout=60)
    
    # 启用hf_transfer加速下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # 逐个下载
    for i, model in enumerate(MODELS, 1):
        log(f"\n--- [{i}/{len(MODELS)}] {model['name']} ({model['desc']}) ---")
        try:
            download_model(model["id"])
        except Exception as e:
            log(f"[ERROR] {model['id']}: {e}")
    
    # 检查下载结果
    log("\n" + "=" * 55)
    log("下载完成！检查文件：")
    for i, model in enumerate(MODELS, 1):
        dir_name = model["id"].replace("/", "--")
        path = f"{MODEL_CACHE_DIR}/{dir_name}"
        if os.path.exists(path):
            size = sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")) / 1e9
            log(f"  [OK] {model['name']}: {size:.2f}GB — {path}")
        else:
            log(f"  [MISS] {model['name']}")
    
    total_size = sum(os.path.getsize(f"{MODEL_CACHE_DIR}/{f}") for f in os.listdir(MODEL_CACHE_DIR) if os.path.isfile(f"{MODEL_CACHE_DIR}/{f}")) / 1e9
    log(f"\n总计: {total_size:.2f}GB")
    log(f"\n下一步: 在Kaggle → Output → Save as Dataset")
    log(f"Dataset名称建议: kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
