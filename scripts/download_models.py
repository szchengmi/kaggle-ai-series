#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v9
====================
根据代码实际加载方式下载：
- SD 1.5: 完整目录 (StableDiffusionPipeline.from_pretrained)
- AnimateDiff: 核心权重文件 (MotionAdapter.from_pretrained)
- Qwen2.5-3B: 完整目录 (AutoModelForCausalLM.from_pretrained)
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

def download_with_hf(model_id, target, include_patterns):
    """用hf download下载指定pattern的文件"""
    os.makedirs(target, exist_ok=True)
    patterns = " ".join([f'"{p}"' for p in include_patterns])
    cred = f' --token {HF_TOKEN}' if HF_TOKEN else ''
    cmd = f'hf download {model_id} --include {patterns} --local-dir {target}{cred}'
    r = run_cmd(cmd, timeout=900)
    return r.returncode == 0

def download_with_git(model_id, target):
    """用git clone下载完整仓库（支持HF Token认证）"""
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)
    
    if HF_TOKEN:
        # 用credential helper方式认证，避免token嵌入URL导致需要password
        os.environ["GIT_TERMINAL_PROMPT"] = "0"
        cred_cmd = f'credential.helper=store --file=/tmp/.git-credentials'
        repo_url = f"https://huggingface.co/{model_id}"
        
        # 先写入credentials文件
        with open("/tmp/.git-credentials", "w") as cf:
            cf.write(f"https://{HF_TOKEN}@huggingface.co\n")
        
        env = os.environ.copy()
        r = subprocess.run(
            f"git -c {cred_cmd} clone --depth 1 {repo_url} {target}",
            shell=True, capture_output=True, text=True, timeout=900, env=env
        )
    else:
        repo_url = f"https://huggingface.co/{model_id}"
        r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=900)
    
    return r.returncode == 0

def is_model_ready(model_id, dir_name, min_size_mb=100):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    for f in os.listdir(target):
        fp = f"{target}/{f}"
        if f.endswith(('.safetensors', '.bin', '.gguf')) and os.path.isfile(fp):
            if os.path.getsize(fp) > min_size_mb * 1024 * 1024:
                return True
    return False


# ============================================================
# 模型定义
# ============================================================

MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "dir": "stable-diffusion-v1-5",
        "type": "full",  # 完整目录，diffusers from_pretrained需要
        "desc": "~2.43GB (完整pipeline)",
    },
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff",
        "dir": "animatediff",
        "type": "core",  # 只下核心权重+config
        "desc": "~301MB (核心权重)",
        "files": [
            "diffusion_pytorch_model.safetensors",
            "config.json",
        ],
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "dir": "Qwen2.5-3B-Instruct",
        "type": "full",  # 完整目录，transformers from_pretrained需要
        "desc": "~6.44GB (完整模型)",
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v9")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")

        target = f"{MODEL_CACHE_DIR}/{model['dir']}"

        if is_model_ready(model["id"], model["dir"]):
            size = get_dir_size_gb(target)
            log(f"  ✅ 已存在 ({size:.2f}GB)")
            check_capacity(f"#{i} ")
            continue

        t0 = time.time()

        if model["type"] == "full":
            # 完整目录 — 用git clone（不需要临时目录，省空间）
            log(f"  ⬇️  git clone (完整目录)")
            ok = download_with_git(model["id"], target)
        else:
            # 核心文件 — 用hf download指定文件
            log(f"  ⬇️  hf download (核心文件)")
            ok = download_with_hf(model["id"], target, model["files"])

            if not ok:
                # fallback: git clone完整仓库
                log(f"  ⚠️  hf失败，降级git clone")
                ok = download_with_git(model["id"], target)

        elapsed = time.time() - t0
        size = get_dir_size_gb(target)

        if ok and size > 0.1:
            log(f"  ✅ {model['name']} ({size:.2f}GB, {elapsed:.0f}秒)")
        else:
            log(f"  ❌ {model['name']} 失败")

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
