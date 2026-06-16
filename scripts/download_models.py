#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v2
====================
下载模型到Kaggle Output目录，确保能被Save as Dataset追踪。
"""

import os
import sys
import time
import subprocess

# ============================================================
# 关键：Kaggle只追踪 /kaggle/working/ 下的输出
# ============================================================
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

def check_disk_space():
    """检查磁盘空间"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/kaggle/working")
        log(f"磁盘: 总计{total/1e9:.1f}GB 已用{used/1e9:.1f}GB 可用{free/1e9:.1f}GB")
        return free
    except:
        return None

def download_model(model_id, dir_name=None):
    """用Python huggingface_hub API下载模型"""
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if os.path.exists(target):
        files = [f for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")]
        size = sum(os.path.getsize(f"{target}/{f}") for f in files) / 1e9
        if size > 0.1:
            log(f"[SKIP] {dir_name} ({size:.2f}GB, {len(files)} files)")
            return True

    log(f"[DOWNLOAD] {model_id}")
    os.makedirs(target, exist_ok=True)

    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            local_dir_use_symlinks=False,
        )
        files = [f for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")]
        size = sum(os.path.getsize(f"{target}/{f}") for f in files) / 1e9
        log(f"[OK] {dir_name} ({size:.2f}GB, {len(files)} files)")
        return True
    except Exception as e:
        log(f"[FAIL] {model_id}: {e}")
        # fallback: git clone
        try:
            log(f"[RETRY] git clone {model_id}")
            repo_url = f"https://huggingface.co/{model_id}"
            if HF_TOKEN:
                repo_url = repo_url.replace("https://", f"https://{HF_TOKEN}@")
            r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=600)
            if r.returncode == 0:
                files = [f for f in os.listdir(target) if os.path.isfile(f"{target}/{f}")]
                size = sum(os.path.getsize(f"{target}/{f}") for f in files) / 1e9
                log(f"[OK] {dir_name} via git ({size:.2f}GB)")
                return True
        except Exception as e2:
            log(f"[FAIL] git: {e2}")
        return False


# ============================================================
# 模型列表
# ============================================================

MODELS = [
    {"id": "runwayml/stable-diffusion-v1-5", "name": "SD 1.5", "desc": "~2.43GB"},
    {"id": "stabilityai/sd-vae-ft-mse", "name": "VAE", "desc": "~334MB"},
    {"id": "guoyww/animatediff-motion-adapter-v1-5-2", "name": "AnimateDiff", "desc": "~301MB"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "desc": "~6.44GB"},
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v2")
    log("=" * 55)

    check_disk_space()

    log(f"目标: {MODEL_CACHE_DIR}")
    log(f"模型: {len(MODELS)} 个")
    log("=" * 55)

    # 安装 huggingface_hub
    log("安装 huggingface_hub...")
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    ok_count = 0
    for i, model in enumerate(MODELS, 1):
        log(f"\n--- [{i}/{len(MODELS)}] {model['name']} ({model['desc']}) ---")
        if download_model(model["id"]):
            ok_count += 1
        check_disk_space()

    # 最终检查
    log("\n" + "=" * 55)
    log("下载结果：")
    total_size = 0
    for model in MODELS:
        dir_name = model["id"].replace("/", "--")
        path = f"{MODEL_CACHE_DIR}/{dir_name}"
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]
            size = sum(os.path.getsize(f"{path}/{f}") for f in files) / 1e9
            total_size += size
            status = "[OK]" if size > 0.1 else "[EMPTY]"
            log(f"  {status} {model['name']}: {size:.2f}GB ({len(files)} files) → {path}")
        else:
            log(f"  [MISS] {model['name']}")

    log(f"\n总计: {total_size:.2f}GB ({ok_count}/{len(MODELS)} 成功)")

    # 列出output目录结构（方便确认Save Dataset的内容）
    log(f"\nOutput目录结构：")
    if os.path.exists("/kaggle/working/output"):
        for item in sorted(os.listdir("/kaggle/working/output")):
            full = f"/kaggle/working/output/{item}"
            if os.path.isdir(full):
                sub = os.listdir(full)
                log(f"  📁 {item}/ ({len(sub)} items)")
                for s in sub[:5]:
                    log(f"      └── {s}")
                if len(sub) > 5:
                    log(f"      └── ... 还有{len(sub)-5}个")
            else:
                log(f"  📄 {item}")

    log(f"\n✅ 完成！现在可以：")
    log(f"   1. Kaggle页面 → Output → Save as Dataset")
    log(f"   2. 名称填: kaggle-ai-series-models")
    log(f"   3. 内容: /kaggle/working/output/models/ 下所有文件夹")
    log("=" * 55)


if __name__ == "__main__":
    main()
