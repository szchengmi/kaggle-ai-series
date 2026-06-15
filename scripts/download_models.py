#!/usr/bin/env python3
"""
Kaggle 模型下载脚本
====================
一次性下载所有需要的模型到 /kaggle/working/output/models/
下载完成后保存到Kaggle Dataset，后续直接挂载使用。
"""

import os
import sys
import time
import subprocess

MODEL_CACHE_DIR = "/kaggle/working/output/models"

# HuggingFace Token
def get_kaggle_secret(key_name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass
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
    print("[WARN] HF_TOKEN 未设置")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def download_model(model_id, dir_name=None):
    """用Python huggingface_hub API下载模型"""
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if os.path.exists(target) and len(os.listdir(target)) > 0:
        size = sum(os.path.getsize(f"{target}/{f}") for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")) / 1e9
        if size > 0.1:
            log(f"[SKIP] {model_id} ({size:.2f}GB 已存在)")
            return

    log(f"[DOWNLOAD] {model_id} → {dir_name}")
    os.makedirs(target, exist_ok=True)

    # 用Python API下载（避免huggingface-cli的HF_HUB_ENABLE_HF_TRANSFER warning）
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        size = sum(os.path.getsize(f"{target}/{f}") for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")) / 1e9
        log(f"[OK] {model_id} ({size:.2f}GB)")
    except Exception as e:
        log(f"[FAIL] {model_id}: {e}")
        # fallback: git clone
        log(f"[RETRY] 用git clone下载 {model_id}")
        try:
            repo_url = f"https://huggingface.co/{model_id}"
            if HF_TOKEN:
                repo_url = repo_url.replace("https://", f"https://{HF_TOKEN}@")
            r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=600)
            if r.returncode == 0:
                size = sum(os.path.getsize(f"{target}/{f}") for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")) / 1e9
                log(f"[OK] {model_id} via git ({size:.2f}GB)")
            else:
                log(f"[FAIL] git clone也失败: {r.stderr[:200]}")
        except Exception as e2:
            log(f"[FAIL] git clone异常: {e2}")


# ============================================================
# 模型列表
# ============================================================

MODELS = [
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
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff Motion",
        "desc": "AnimateDiff Motion Adapter (301MB)"
    },
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
    log(f"目标: {MODEL_CACHE_DIR}")
    log(f"模型: {len(MODELS)} 个")
    log("=" * 55)

    # 安装 huggingface_hub
    log("安装 huggingface_hub...")
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n--- [{i}/{len(MODELS)}] {model['name']} ({model['desc']}) ---")
        try:
            download_model(model["id"])
        except Exception as e:
            log(f"[ERROR] {model['id']}: {e}")

    # 检查结果
    log("\n" + "=" * 55)
    log("下载完成！检查文件：")
    total_size = 0
    for model in MODELS:
        dir_name = model["id"].replace("/", "--")
        path = f"{MODEL_CACHE_DIR}/{dir_name}"
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]
            size = sum(os.path.getsize(f"{path}/{f}") for f in files) / 1e9
            total_size += size
            log(f"  {'[OK]' if size > 0.1 else '[EMPTY]'} {model['name']}: {size:.2f}GB ({len(files)} files)")
        else:
            log(f"  [MISS] {model['name']}")

    log(f"\n总计: {total_size:.2f}GB")
    log(f"\n保存Dataset: Kaggle页面 → Output → Save as Dataset")
    log(f"名称: kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
