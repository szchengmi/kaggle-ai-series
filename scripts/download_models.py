#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v11
====================
用huggingface_hub Python API直接下载（不走代理，Kaggle能直连hf.co）
逐个下载，实时容量监控，断点续传
"""

import os
import sys
import time
import shutil

MODEL_CACHE_DIR = "/kaggle/working/output/models"

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
    print("[OK] HF_TOKEN")
else:
    print("[WARN] HF_TOKEN 未设置")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_disk_free_gb():
    import shutil as _s
    _, _, free = _s.disk_usage("/kaggle/working")
    return free / 1e9

def get_dir_size_gb(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
            except:
                pass
    return total / 1e9

def check_capacity(label=""):
    free = get_disk_free_gb()
    used = get_dir_size_gb(MODEL_CACHE_DIR)
    log(f"📊 {label}剩余: {free:.1f}GB | models: {used:.2f}GB")
    return free

def is_model_ready(dir_name, min_size_mb=100):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    for f in os.listdir(target):
        fp = f"{target}/{f}"
        if f.endswith(('.safetensors', '.bin', '.gguf')) and os.path.isfile(fp):
            if os.path.getsize(fp) > min_size_mb * 1024 * 1024:
                return True
    return False

def download_model(model_id, dir_name, allow_patterns=None):
    """用huggingface_hub snapshot_download直接下载"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if is_model_ready(dir_name):
        size = get_dir_size_gb(target)
        log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
        return True

    log(f"  ⬇️  {model_id}")
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        elapsed = time.time() - t0
        size = get_dir_size_gb(target)
        if size > 0.1:
            log(f"  ✅ {dir_name} ({size:.2f}GB, {elapsed:.0f}秒)")
            return True
        else:
            log(f"  ❌ 下载为空")
            shutil.rmtree(target, ignore_errors=True)
            return False
    except Exception as e:
        log(f"  ❌ {e}")
        shutil.rmtree(target, ignore_errors=True)
        return False


# ============================================================
# 模型定义
# ============================================================

MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "dir": "stable-diffusion-v1-5",
        "desc": "~2.43GB (完整pipeline)",
        "patterns": None,
    },
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff",
        "dir": "animatediff",
        "desc": "~301MB (核心权重)",
        "patterns": ["diffusion_pytorch_model.safetensors", "config.json"],
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "dir": "Qwen2.5-3B-Instruct",
        "desc": "~6.44GB (完整模型)",
        "patterns": None,
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v11")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    import subprocess
    subprocess.run("pip install -q -U huggingface_hub", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")
        download_model(model["id"], model["dir"], model["patterns"])
        check_capacity(f"#{i} ")

    # 最终结果
    log(f"\n{'='*55}")
    log("全部完成！")
    total = get_dir_size_gb(MODEL_CACHE_DIR)
    free = get_disk_free_gb()
    log(f"模型总计: {total:.2f}GB | 磁盘剩余: {free:.1f}GB")

    for model in MODELS:
        path = f"{MODEL_CACHE_DIR}/{model['dir']}"
        size = get_dir_size_gb(path)
        done = "✅" if size > 0.1 else "❌"
        log(f"  {done} {model['name']}: {size:.2f}GB")

    log(f"\n✅ Save as Dataset → kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
