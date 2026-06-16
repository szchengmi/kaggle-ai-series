#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v6
====================
用git clone下载（不需要临时目录，节省50%空间）
逐个下载，每个下载完清理+检查容量
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
    log(f"📊 {label}磁盘剩余: {free:.1f}GB | models已用: {used:.2f}GB")
    return free

def is_model_complete(model_id, dir_name):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    for f in os.listdir(target):
        fp = f"{target}/{f}"
        if f.endswith(('.safetensors', '.bin')) and os.path.isfile(fp):
            if os.path.getsize(fp) > 50 * 1024 * 1024:
                return True
    return False

def download_model_git(model_id, dir_name):
    """用git clone下载 — 不需要临时目录，省50%空间"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if is_model_complete(model_id, dir_name):
        size = get_dir_size_gb(target)
        log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
        return True

    # 清理旧的不完整下载
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors=True)

    log(f"  ⬇️  git clone {model_id}")
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    # 构建URL
    repo_url = f"https://huggingface.co/{model_id}"
    if HF_TOKEN:
        # git credential helper方式
        cred_url = f"https://{HF_TOKEN}@huggingface.co/{model_id}"
    else:
        cred_url = repo_url

    # git clone --depth 1（只下载最新commit，不下载历史）
    env = os.environ.copy()
    if HF_TOKEN:
        # 用header方式认证（更可靠）
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_SYSTEM"] = "/dev/null"

    r = run_cmd(
        f'git -c "http.extraHeader=Cookie: token={HF_TOKEN}" clone --depth 1 {repo_url} {target}',
        timeout=900
    )

    if r.returncode != 0 and HF_TOKEN:
        # fallback: 用header认证
        log(f"  🔄 重试认证方式...")
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        r = run_cmd(
            f'git -c "http.extraHeader=Authorization: Bearer {HF_TOKEN}" clone --depth 1 {repo_url} {target}',
            timeout=900
        )

    if r.returncode != 0:
        log(f"  ⚠️  git clone失败，尝试无认证...")
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=900)

    elapsed = time.time() - t0

    if r.returncode == 0 and is_model_complete(model_id, dir_name):
        size = get_dir_size_gb(target)
        log(f"  ✅ {dir_name} ({size:.2f}GB, {elapsed:.0f}秒)")
        return True
    else:
        log(f"  ❌ {dir_name} 下载失败")
        return False


# ============================================================
# 模型列表
# ============================================================

MODELS = [
    {"id": "runwayml/stable-diffusion-v1-5", "name": "SD 1.5", "size": "~2.43GB"},
    {"id": "guoyww/animatediff-motion-adapter-v1-5-2", "name": "AnimateDiff", "size": "~301MB"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "size": "~6.44GB"},
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v6 (git clone)")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['size']})")

        dir_name = model["id"].replace("/", "--")
        ok = download_model_git(model["id"], dir_name)
        check_capacity(f"下载{i}后 ")

        if not ok:
            log(f"  ⚠️  {model['name']} 失败，继续下一个...")

    # 最终结果
    log(f"\n{'='*55}")
    log("全部完成！")
    total = get_dir_size_gb(MODEL_CACHE_DIR)
    free = get_disk_free_gb()
    log(f"模型总计: {total:.2f}GB | 磁盘剩余: {free:.1f}GB")

    for model in MODELS:
        dir_name = model["id"].replace("/", "--")
        size = get_dir_size_gb(f"{MODEL_CACHE_DIR}/{dir_name}")
        done = "✅" if size > 0.1 else "❌"
        log(f"  {done} {model['name']}: {size:.2f}GB")

    log(f"\n✅ Save as Dataset → kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
