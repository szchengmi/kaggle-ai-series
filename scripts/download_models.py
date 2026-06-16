#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v8
====================
只下载核心模型文件（模型权重+文本编码器+VAE）
像ComfyUI一样，每个模型只需要3个文件
"""

import os
import sys
import time
import shutil
import subprocess

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

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=900):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def get_disk_free_gb():
    import shutil
    _, _, free = shutil.disk_usage("/kaggle/working")
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

def download_files(model_id, target, files):
    """用hf下载指定文件列表"""
    from huggingface_hub import hf_hub_download
    os.makedirs(target, exist_ok=True)
    ok = 0
    for f in files:
        try:
            dest_dir = os.path.dirname(f"{target}/{f}")
            os.makedirs(dest_dir, exist_ok=True)
            hf_hub_download(repo_id=model_id, filename=f, local_dir=target, local_dir_use_symlinks=False)
            ok += 1
            log(f"    ✅ {f}")
        except Exception as e:
            log(f"    ❌ {f}: {e}")
    return ok

def download_model(model_id, dir_name, core_files):
    """下载模型的核心文件"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    # 检查是否已完成
    if os.path.exists(target):
        existing = [f for f in os.listdir(target) if f.endswith(('.safetensors', '.bin')) and os.path.isfile(f"{target}/{f}")]
        if existing:
            size = get_dir_size_gb(target)
            log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB, {len(existing)}个文件)")
            return True

    log(f"  ⬇️  {model_id} ({len(core_files)}个核心文件)")
    t0 = time.time()

    ok = download_files(model_id, target, core_files)
    elapsed = time.time() - t0
    size = get_dir_size_gb(target)

    if ok > 0:
        log(f"  ✅ {dir_name} ({size:.2f}GB, {ok}/{len(core_files)}个文件, {elapsed:.0f}秒)")
        return True
    else:
        log(f"  ❌ 下载失败")
        return False


# ============================================================
# 模型列表 — 只列核心文件
# ============================================================

MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "dir": "stable-diffusion-v1-5",
        "files": [
            # 模型核心 (UNet + 内置text_encoder + 内置vae 全在一个文件里)
            "v1-5-pruned-emaonly.safetensors",
            # 配置文件
            "config.json",
        ],
        # 如果上面那个文件不存在，用这些
        "alt_files": [
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "config.json",
        ],
    },
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff",
        "dir": "animatediff-motion-adapter-v1-5-2",
        "files": [
            # Motion adapter核心权重
            "v1-5-2.ckpt",
            # 配置文件
            "config.json",
        ],
        "alt_files": [
            "diffusion_pytorch_model.safetensors",
            "config.json",
        ],
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "dir": "Qwen2.5-3B-Instruct",
        "files": [
            # 模型权重（分片）
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            # 配置文件
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ],
        "alt_files": [],
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v8 (核心文件)")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    # 安装huggingface_hub
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']}")

        ok = download_model(model["id"], model["dir"], model["files"])

        # 如果主文件列表失败，尝试alt
        if not ok and model.get("alt_files"):
            log(f"  🔄 尝试备用文件列表...")
            target = f"{MODEL_CACHE_DIR}/{model['dir']}"
            shutil.rmtree(target, ignore_errors=True)
            download_model(model["id"], model["dir"], model["alt_files"])

        check_capacity(f"下载{i}后 ")

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
