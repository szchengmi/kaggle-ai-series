#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v3
====================
- 检测已下载的文件，跳过完成的模型
- 支持断点续传（resume_download）
- 实时显示下载速度和进度
- 下载完验证文件大小
"""

import os
import sys
import time
import subprocess
import json

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
    except:
        pass
    return ""

HF_TOKEN = get_kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("[OK] HF_TOKEN 已设置")
else:
    print("[WARN] HF_TOKEN 未设置，限速下载")

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
    """获取目录总大小（GB）"""
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
    """模型期望大小（GB）"""
    sizes = {
        "runwayml/stable-diffusion-v1-5": 2.43,
        "stabilityai/sd-vae-ft-mse": 0.33,
        "guoyww/animatediff-motion-adapter-v1-5-2": 0.30,
        "Qwen/Qwen2.5-3B-Instruct": 6.44,
    }
    return sizes.get(model_id, 0)

def load_progress():
    """加载下载进度"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def is_model_complete(model_id, dir_name):
    """检查模型是否已完整下载"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    
    current_size = get_dir_size(target)
    expected = get_expected_size(model_id)
    
    # 已下载超过期望大小的90%算完成
    if expected > 0 and current_size >= expected * 0.9:
        return True
    
    # 检查是否有.safetensors或.bin文件（模型核心文件）
    has_model_files = False
    for f in os.listdir(target):
        if f.endswith(('.safetensors', '.bin', '.gguf', '.pt')) and os.path.isfile(f"{target}/{f}"):
            fsize = os.path.getsize(f"{target}/{f}")
            if fsize > 10 * 1024 * 1024:  # >10MB
                has_model_files = True
                break
    
    return has_model_files and current_size > 0.5

def download_model(model_id, dir_name=None):
    """下载模型，支持断点续传"""
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    # 检查是否已完成
    if is_model_complete(model_id, dir_name):
        size = get_dir_size(target)
        log(f"[SKIP] {dir_name} ({size:.2f}GB 已下载)")
        return True

    # 检查已有进度
    progress = load_progress()
    prev_time = progress.get(dir_name, {}).get("time", "")
    prev_size = progress.get(dir_name, {}).get("size_gb", 0)
    if prev_size > 0:
        current_size = get_dir_size(target)
        if current_size > prev_size * 0.95:
            log(f"[RESUME] {dir_name} (之前已下载{prev_size:.2f}GB，当前{current_size:.2f}GB)")
        else:
            log(f"[RETRY] {dir_name} (之前下载不完整：{prev_size:.2f}GB)")

    log(f"[DOWNLOAD] {model_id}")
    os.makedirs(target, exist_ok=True)

    from huggingface_hub import snapshot_download
    try:
        t0 = time.time()
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            resume_download=True,  # 关键：启用断点续传
        )
        elapsed = time.time() - t0
        size = get_dir_size(target)
        speed = size / elapsed * 1e9 / 1e6 if elapsed > 0 else 0  # MB/s
        
        # 保存进度
        progress[dir_name] = {"size_gb": round(size, 2), "time": time.strftime("%Y-%m-%d %H:%M")}
        save_progress(progress)
        
        log(f"[OK] {dir_name} ({size:.2f}GB, {elapsed:.0f}秒, {speed:.0f}MB/s)")
        return True
    except Exception as e:
        log(f"[FAIL] {model_id}: {e}")
        # 保存当前进度
        current_size = get_dir_size(target)
        if current_size > 0:
            progress[dir_name] = {"size_gb": round(current_size, 2), "time": time.strftime("%Y-%m-%d %H:%M")}
            save_progress(progress)
            log(f"[SAVE] 已保存进度 {current_size:.2f}GB，下次运行会续传")
        
        # fallback: git clone
        try:
            log(f"[RETRY] 尝试 git clone {model_id}")
            # 清理不完整的目录
            import shutil
            if os.path.exists(target):
                shutil.rmtree(target)
            os.makedirs(target, exist_ok=True)
            
            repo_url = f"https://huggingface.co/{model_id}"
            if HF_TOKEN:
                repo_url = repo_url.replace("https://", f"https://{HF_TOKEN}@")
            r = run_cmd(f"git clone --depth 1 {repo_url} {target}", timeout=600)
            if r.returncode == 0:
                size = get_dir_size(target)
                progress[dir_name] = {"size_gb": round(size, 2), "time": time.strftime("%Y-%m-%d %H:%M")}
                save_progress(progress)
                log(f"[OK] {dir_name} via git ({size:.2f}GB)")
                return True
        except Exception as e2:
            log(f"[FAIL] git clone: {e2}")
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
    log("  Kaggle AI短剧 - 模型下载 v3")
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
        size = get_dir_size(path)
        total_size += size
        expected = get_expected_size(model["id"])
        pct = (size / expected * 100) if expected > 0 else 0
        status = "✅" if size >= expected * 0.9 else ("⚠️" if size > 0.1 else "❌")
        log(f"  {status} {model['name']}: {size:.2f}GB / {expected:.2f}GB ({pct:.0f}%) → {path}")

    log(f"\n总计: {total_size:.2f}GB ({ok_count}/{len(MODELS)} 完成)")

    # Output目录结构
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

    log(f"\n✅ 完成！")
    log(f"   Save as Dataset → 名称: kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
