#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v12
====================
用aria2c直接下载每个模型的文件（不经过.cache临时目录）
逐个下载，实时容量监控
"""

import os
import sys
import time
import shutil
import subprocess
import json

MODEL_CACHE_DIR = "/kaggle/working/output/models"
PROGRESS_FILE = f"{MODEL_CACHE_DIR}/.download_progress.json"

def get_kaggle_secret(key_name):
    # 方式1: kaggle_secrets库
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(key_name)
        if val:
            return val
    except:
        pass
    # 方式2: Kaggle自动注入的变量
    try:
        if key_name == "HF_TOKEN":
            return secret_value_1  # noqa: F821
        if key_name == "GOOGLE_API_KEY":
            return secret_value_0  # noqa: F821
    except:
        pass
    # 方式3: 环境变量
    return os.environ.get(key_name, "")

HF_TOKEN = get_kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print(f"[OK] HF_TOKEN: {HF_TOKEN[:10]}...")
else:
    print("[WARN] HF_TOKEN 未设置")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=300):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

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

def get_repo_file_list(model_id):
    """列出仓库中所有文件"""
    from huggingface_hub import list_repo_tree
    try:
        files = []
        for f in list_repo_tree(repo_id=model_id, repo_type="model"):
            if hasattr(f, 'path'):
                files.append(f.path)
            elif hasattr(f, 'rfilename'):
                files.append(f.rfilename)
            else:
                files.append(str(f))
        return files
    except Exception as e:
        log(f"  ⚠️  获取文件列表失败: {e}")
        return []

def filter_core_files(all_files, model_id=None):
    """过滤只保留核心模型文件"""
    keep = []
    for f in all_files:
        # 跳过无关文件
        if any(x in f for x in ['.git', 'tests/', 'test_', '.github', 'LICENSE', 'README.md',
                                  '.gitattributes', '.gitignore', 'Makefile']):
            continue
        # SD 1.5特殊处理：只保留ema-only版本（4GB），不要完整未剪枝版（7.7GB）
        if model_id == "runwayml/stable-diffusion-v1-5":
            if f == "v1-5-pruned-emaonly.safetensors":
                keep.append(f)
                continue
            if f == "v1-5-pruned.safetensors":
                continue  # 跳过7.7GB的完整未剪枝版
        # 保留模型核心文件
        if any(f.endswith(p) for p in ['.safetensors', '.bin', '.gguf', '.pt', '.pth',
                                       'config.json', 'tokenizer.json', 'tokenizer_config.json',
                                       'special_tokens_map.json', 'merges.txt', 'vocab.json',
                                       'generation_config.json', 'preprocessor_config.json']):
            keep.append(f)
    return keep

def download_file_aria2(url, dest_path):
    """用aria2c下载单个文件"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cmd = f'aria2c -x 4 -s 4 -k 1M --async-dns=false -o "{os.path.basename(dest_path)}" -d "{os.path.dirname(dest_path)}" "{url}"'
    r = run_cmd(cmd, timeout=300)
    return r.returncode == 0

def download_file_wget(url, dest_path):
    """用wget下载单个文件"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cmd = f'wget -q --show-progress -O "{dest_path}" "{url}"'
    r = subprocess.run(cmd, shell=True, timeout=300)
    return r.returncode == 0

def download_file_hf_api(model_id, filename, dest_path):
    """用huggingface_hub hf_hub_download下载单文件，直接到目标位置"""
    from huggingface_hub import hf_hub_download
    try:
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        # 关键：用cache_dir直接指向最终目录，避免.cache临时目录
        result = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=dest_dir,
            cache_dir=dest_dir,
        )
        return True
    except Exception as e:
        log(f"    fail {filename}: {e}")
        return False

def download_model(model_id, dir_name):
    """逐个文件下载模型"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if is_model_ready(dir_name):
        size = get_dir_size_gb(target)
        log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
        return True

    log(f"  ⬇️  {model_id}")
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    # 获取文件列表
    all_files = get_repo_file_list(model_id)
    if not all_files:
        log(f"  ❌ 无法获取文件列表")
        return False

    # 过滤核心文件
    core_files = filter_core_files(all_files, model_id)
    log(f"  文件: {len(all_files)}个 → 下载{len(core_files)}个核心文件")
    for cf in core_files:
        log(f"    → {cf}")

    # 逐个文件下载
    ok = 0
    fail = 0
    base_url = f"https://huggingface.co/{model_id}/resolve/main"

    for f in core_files:
        dest = f"{target}/{f}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            ok += 1
            continue

        url = f"{base_url}/{f}"

        # 方式1: hf_hub_download
        if download_file_hf_api(model_id, f, dest):
            ok += 1
        else:
            # 方式2: aria2c
            if download_file_aria2(url, dest):
                ok += 1
            else:
                fail += 1

        # 每个文件后检查容量
        free = get_disk_free_gb()
        if free < 1:
            log(f"  ⚠️  磁盘空间不足！剩余{free:.1f}GB")
            break

    elapsed = time.time() - t0
    size = get_dir_size_gb(target)

    if ok > 0 and size > 0.1:
        log(f"  ✅ {dir_name} ({size:.2f}GB, {ok}/{len(core_files)}个文件, {elapsed:.0f}秒)")
        return True
    else:
        log(f"  ❌ 失败 ({fail}个文件失败)")
        return False


# ============================================================
# 模型定义
# ============================================================

MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "dir": "stable-diffusion-v1-5",
        "desc": "~2.43GB",
    },
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff",
        "dir": "animatediff",
        "desc": "~301MB",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "dir": "Qwen2.5-3B-Instruct",
        "desc": "~6.44GB",
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v12 (aria2/逐文件)")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    # 安装huggingface_hub
    subprocess.run("pip install -q -U huggingface_hub aria2", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")
        download_model(model["id"], model["dir"])
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
