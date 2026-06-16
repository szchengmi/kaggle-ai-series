#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v7
====================
只下载核心模型文件（.safetensors/.bin），不下载README/git历史等
用huggingface-cli指定文件列表下载
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
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
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

def list_repo_files(model_id):
    """列出仓库中的文件"""
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(repo_id=model_id, repo_type="model")
        return files
    except Exception as e:
        log(f"  ⚠️  列出文件失败: {e}")
        return []

def filter_model_files(files):
    """只保留核心模型文件"""
    keep = []
    patterns = ['.safetensors', '.bin', '.gguf', '.pt', '.pth',
                'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'merges.txt', 'vocab.json',
                'preprocessor_config.json', 'scheduler_config.json',
                'model_index.json', 'pipeline_config.json']
    for f in files:
        # 跳过git文件、测试文件、大无关文件
        if any(skip in f for skip in ['.git', 'tests/', 'test_', '.github', 'LICENSE', 'README']):
            continue
        if any(f.endswith(p) for p in patterns):
            keep.append(f)
        # 也保留vae/scheduler子目录的配置
        if '/vae/' in f or '/scheduler/' in f or '/text_encoder/' in f:
            if not any(skip in f for skip in ['.git', 'tests/']):
                keep.append(f)
    return keep

def download_model(model_id, dir_name=None):
    if dir_name is None:
        dir_name = model_id.replace("/", "--")
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    # 检查是否已完成
    if os.path.exists(target):
        size = get_dir_size_gb(target)
        for f in os.listdir(target):
            if f.endswith(('.safetensors', '.bin')) and os.path.isfile(f"{target}/{f}"):
                if os.path.getsize(f"{target}/{f}") > 50 * 1024 * 1024:
                    log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
                    return True

    log(f"  ⬇️  {model_id}")
    t0 = time.time()

    # 列出并过滤文件
    files = list_repo_files(model_id)
    if not files:
        log(f"  ❌ 无法获取文件列表")
        return False

    keep_files = filter_model_files(files)
    log(f"  文件: {len(files)}个 → 只下载{len(keep_files)}个核心文件")

    # 创建临时目录
    tmp_dir = f"{target}_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # 用huggingface-cli下载指定文件
    for f in keep_files:
        # 创建子目录
        sub_dir = os.path.dirname(f"{tmp_dir}/{f}")
        os.makedirs(sub_dir, exist_ok=True)

    # 批量下载（用include参数）
    include_pattern = " ".join([f'"{f}"' for f in keep_files[:20]])
    if len(keep_files) > 20:
        # 太多文件，用pattern方式
        include_pattern = '"*.safetensors" "*.bin" "config.json" "tokenizer.json" "tokenizer_config.json"'

    cred = f' --token {HF_TOKEN}' if HF_TOKEN else ''
    cmd = f'hf download {model_id} --include {include_pattern} --local-dir {tmp_dir}{cred}'

    log(f"  下载中...")
    r = run_cmd(cmd, timeout=900)

    elapsed = time.time() - t0

    if r.returncode == 0 and os.path.exists(tmp_dir):
        # 检查是否有模型文件
        has_model = False
        for dirpath, _, filenames in os.walk(tmp_dir):
            for fn in filenames:
                if fn.endswith(('.safetensors', '.bin')) and os.path.isfile(f"{dirpath}/{fn}"):
                    if os.path.getsize(f"{dirpath}/{fn}") > 50 * 1024 * 1024:
                        has_model = True
                        break

        if has_model:
            # 移动到最终位置
            if os.path.exists(target):
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(tmp_dir, target)
            size = get_dir_size_gb(target)
            log(f"  ✅ {dir_name} ({size:.2f}GB, {elapsed:.0f}秒)")
            return True
        else:
            log(f"  ❌ 下载的文件中没有模型权重")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False
    else:
        log(f"  ⚠️  huggingface-cli失败: {r.stderr[:100] if r.stderr else ''}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # fallback: 直接用Python API下载单个文件
        log(f"  🔄 尝试逐文件下载...")
        return download_files_fallback(model_id, target, keep_files, dir_name)


def download_files_fallback(model_id, target, files, dir_name):
    """逐文件下载fallback"""
    from huggingface_hub import hf_hub_download
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    ok_count = 0
    for f in files:
        try:
            dest = f"{target}/{f}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            hf_hub_download(repo_id=model_id, filename=f, local_dir=target)
            ok_count += 1
        except Exception as e:
            pass  # 跳过失败的文件

    elapsed = time.time() - t0
    size = get_dir_size_gb(target)

    if ok_count > 0:
        log(f"  ✅ {dir_name} ({size:.2f}GB, {ok_count}个文件, {elapsed:.0f}秒)")
        return True
    else:
        log(f"  ❌ 逐文件下载失败")
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
    log("  Kaggle AI短剧 - 模型下载 v7 (只下核心文件)")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    # 安装huggingface_hub
    run_cmd("pip install -q -U huggingface_hub", timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['size']})")

        dir_name = model["id"].replace("/", "--")
        download_model(model["id"], dir_name)
        check_capacity(f"下载{i}后 ")

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
