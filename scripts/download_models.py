#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v4
====================
- 精简模型列表（只下载必要的）
- 支持代理（解决Gemini 403问题）
- 断点续传 + 进度检测
"""

import os
import sys
import time
import json
import subprocess

MODEL_CACHE_DIR = "/kaggle/working/output/models"
PROGRESS_FILE = f"{MODEL_CACHE_DIR}/.download_progress.json"

def get_kaggle_secret(key_name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass
    try:
        if key_name == "HF_TOKEN":
            return hf_token  # noqa: F821
        if key_name == "GOOGLE_API_KEY":
            return secret_value_0  # noqa: F821
        if key_name == "PROXY_URL":
            return secret_value_2  # noqa: F821
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

# 代理
PROXY_URL = get_kaggle_secret("PROXY_URL")
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    print(f"[OK] 代理: {PROXY_URL}")
else:
    print("[INFO] 未设置代理")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def check_disk_space():
    try:
        import shutil
        total, used, free = shutil.disk_usage("/kaggle/working")
        log(f"磁盘: 总计{total/1e9:.1f}GB 已用{used/1e9:.1f}GB 可用{free/1e9:.1f}GB")
        return free
    except:
        return None

def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9

def get_expected_size(model_id):
    sizes = {
        "runwayml/stable-diffusion-v1-5": 2.43,
        "stabilityai/sd-vae-ft-mse": 0.33,
        "guoyww/animatediff-motion-adapter-v1-5-2": 0.30,
        "Qwen/Qwen2.5-3B-Instruct": 6.44,
    }
    return sizes.get(model_id, 0)

def is_model_complete(model_id, dir_name):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    current_size = get_dir_size(target)
    expected = get_expected_size(model_id)
    if expected > 0 and current_size >= expected * 0.9:
        return True
    has_model_files = False
    for f in os.listdir(target):
        if f.endswith(('.safetensors', '.bin', '.gguf', '.pt')) and os.path.isfile(f"{target}/{f}"):
            if os.path.getsize(f"{target}/{f}") > 10 * 1024 * 1024:
                has_model_files = True
                break
    return has_model_files and current_size > 0.5

def download_model(model_id, dir_name=None):
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if is_model_complete(model_id, dir_name):
        size = get_dir_size(target)
        log(f"[SKIP] {dir_name} ({size:.2f}GB 已下载)")
        return True

    log(f"[DOWNLOAD] {model_id}")
    os.makedirs(target, exist_ok=True)

    from huggingface_hub import snapshot_download
    try:
        t0 = time.time()
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            resume_download=True,
        )
        elapsed = time.time() - t0
        size = get_dir_size(target)
        speed = size / elapsed * 1e9 / 1e6 if elapsed > 0 else 0
        log(f"[OK] {dir_name} ({size:.2f}GB, {elapsed:.0f}秒, {speed:.0f}MB/s)")
        return True
    except Exception as e:
        log(f"[FAIL] {model_id}: {e}")
        try:
            import shutil
            if os.path.exists(target):
                shutil.rmtree(target)
            os.makedirs(target, exist_ok=True)
            log(f"[RETRY] git clone {model_id}")
            repo_url = f"https://huggingface.co/{model_id}"
            if HF_TOKEN:
                repo_url = repo_url.replace("https://", f"https://{HF_TOKEN}@")
            r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=600)
            if r.returncode == 0:
                size = get_dir_size(target)
                log(f"[OK] {dir_name} via git ({size:.2f}GB)")
                return True
        except Exception as e2:
            log(f"[FAIL] git: {e2}")
        return False


# ============================================================
# 模型列表（精简版）
# ============================================================

MODELS = [
    # 画面+视频生成（必须）
    {"id": "runwayml/stable-diffusion-v1-5", "name": "SD 1.5", "desc": "~2.43GB"},
    {"id": "guoyww/animatediff-motion-adapter-v1-5-2", "name": "AnimateDiff", "desc": "~301MB"},
    # 本地LLM（Gemini被封时备用）
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "desc": "~6.44GB"},
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v4")
    log("=" * 55)

    check_disk_space()

    log(f"目标: {MODEL_CACHE_DIR}")
    log(f"模型: {len(MODELS)} 个")
    log("=" * 55)

    log("安装 huggingface_hub...")
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    ok_count = 0
    for i, model in enumerate(MODELS, 1):
        log(f"\n--- [{i}/{len(MODELS)}] {model['name']} ({model['desc']}) ---")
        if download_model(model["id"]):
            ok_count += 1
        check_disk_space()

    # 结果
    log("\n" + "=" * 55)
    log("下载结果：")
    total_size = 0
    for model in MODELS:
        dir_name = model["id"].replace("/", "--")
        path = f"{MODEL_CACHE_DIR}/{dir_name}"
        size = get_dir_size(path)
        total_size += size
        expected = get_expected_size(model["id"])
        pct = (size / expected * 100) if expected > 0 else 0
        status = "✅" if size >= expected * 0.9 else ("⚠️" if size > 0.1 else "❌")
        log(f"  {status} {model['name']}: {size:.2f}GB / {expected:.2f}GB ({pct:.0f}%)")

    log(f"\n总计: {total_size:.2f}GB ({ok_count}/{len(MODELS)} 完成)")

    # Output结构
    log(f"\nOutput目录结构：")
    if os.path.exists("/kaggle/working/output"):
        for item in sorted(os.listdir("/kaggle/working/output")):
            full = f"/kaggle/working/output/{item}"
            if os.path.isdir(full):
                sub = sorted(os.listdir(full))
                log(f"  📁 {item}/ ({len(sub)} items)")
                for s in sub[:5]:
                    log(f"      └── {s}")
                if len(sub) > 5:
                    log(f"      └── ... 还有{len(sub)-5}个")
            else:
                log(f"  📄 {item}")

    log(f"\n✅ 完成！Save as Dataset → kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
