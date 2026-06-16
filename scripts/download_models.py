#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v5
====================
- 逐个下载，每个下载完清理.cache临时文件
- 实时计算已用/剩余容量
- 断点续传
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
    print("[OK] HF_TOKEN 已设置")
else:
    print("[WARN] HF_TOKEN 未设置")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def get_disk_free():
    """获取剩余磁盘空间（GB）"""
    try:
        import shutil
        _, _, free = shutil.disk_usage("/kaggle/working")
        return free / 1e9
    except:
        return 0

def get_dir_size(path):
    """获取目录大小（GB）"""
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

def clean_cache(model_dir):
    """清理下载后的.cache临时文件"""
    cache_dir = f"{model_dir}/.cache"
    if os.path.exists(cache_dir):
        size = get_dir_size(cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
        log(f"  清理.cache: 释放 {size:.2f}GB")

    # 清理.huggingface目录
    hf_dir = f"{model_dir}/.huggingface"
    if os.path.exists(hf_dir):
        size = get_dir_size(hf_dir)
        shutil.rmtree(hf_dir, ignore_errors=True)
        if size > 0.01:
            log(f"  清理.huggingface: 释放 {size:.2f}GB")

    # 清理lock文件
    for f in os.listdir(model_dir):
        if f.endswith(".lock") or f.endswith(".tmp"):
            try:
                os.remove(f"{model_dir}/{f}")
            except:
                pass

def check_capacity():
    """检查并打印容量"""
    free = get_disk_free()
    used = get_dir_size(MODEL_CACHE_DIR)
    log(f"📊 磁盘: 可用 {free:.1f}GB | models已用 {used:.2f}GB")
    return free

def is_model_complete(model_id, dir_name):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    # 检查是否有核心模型文件
    for f in os.listdir(target):
        fp = f"{target}/{f}"
        if f.endswith(('.safetensors', '.bin')) and os.path.isfile(fp):
            if os.path.getsize(fp) > 50 * 1024 * 1024:  # >50MB
                return True
    return False

def download_with_git(model_id, target):
    """用git clone下载"""
    repo_url = f"https://huggingface.co/{model_id}"
    if HF_TOKEN:
        repo_url = repo_url.replace("https://", f"https://{HF_TOKEN}@")
    r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=600)
    return r.returncode == 0

def download_model(model_id, dir_name=None):
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    # 检查已完成
    if is_model_complete(model_id, dir_name):
        size = get_dir_size(target)
        log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
        return True

    log(f"  ⬇️  下载 {model_id}")
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    # 方法1: snapshot_download
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            resume_download=True,
        )
        elapsed = time.time() - t0
        clean_cache(target)  # 关键：清理临时文件
        size = get_dir_size(target)
        log(f"  ✅ {dir_name} ({size:.2f}GB, {elapsed:.0f}秒)")
        return True
    except Exception as e:
        log(f"  ⚠️  snapshot失败: {e}")

    # 方法2: git clone
    try:
        log(f"  🔄 尝试 git clone...")
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        t0 = time.time()
        if download_with_git(model_id, target):
            elapsed = time.time() - t0
            clean_cache(target)
            size = get_dir_size(target)
            log(f"  ✅ {dir_name} via git ({size:.2f}GB, {elapsed:.0f}秒)")
            return True
    except Exception as e:
        log(f"  ❌ git clone失败: {e}")

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
    log("  Kaggle AI短剧 - 模型下载 v5")
    log("=" * 55)

    log(f"目标: {MODEL_CACHE_DIR}")
    log(f"模型: {len(MODELS)} 个")

    log("安装 huggingface_hub...")
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['size']})")
        check_capacity()

        if download_model(model["id"]):
            check_capacity()
        else:
            log(f"  ❌ {model['name']} 下载失败")

    # 最终结果
    log(f"\n{'='*55}")
    log("下载完成！")
    total = get_dir_size(MODEL_CACHE_DIR)
    free = get_disk_free()
    log(f"模型总计: {total:.2f}GB")
    log(f"磁盘剩余: {free:.1f}GB")

    for model in MODELS:
        dir_name = model["id"].replace("/", "--")
        path = f"{MODEL_CACHE_DIR}/{dir_name}"
        size = get_dir_size(path)
        done = "✅" if size > 0.1 else "❌"
        log(f"  {done} {model['name']}: {size:.2f}GB")

    log(f"\nOutput目录:")
    for item in sorted(os.listdir("/kaggle/working/output")):
        full = f"/kaggle/working/output/{item}"
        if os.path.isdir(full):
            log(f"  📁 {item}/ ({get_dir_size(full):.2f}GB)")

    log(f"\n✅ Save as Dataset → kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
