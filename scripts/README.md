# Kaggle AI短剧自动生成 - 端到端流水线

在Kaggle Notebook中运行，自动完成AI短剧的全流程生成：

1. **剧本生成** - Gemini API / 本地Qwen2.5-3B / 预置剧本（三级降级）
2. **分镜生成** - 结构化JSON
3. **画面生成** - Stable Diffusion 1.5
4. **视频生成** - AnimateDiff-Lightning
5. **配音生成** - ChatTTS / edge-tts
6. **剪辑合成** - FFmpeg

## Kaggle Notebook 使用

### ⚠️ 注意：不要在已有kaggle-ai-series目录下clone

如果之前已经clone过，先清理：
```python
!rm -rf kaggle-ai-series
```

### Step 1: 克隆仓库（首次）

```python
!git clone https://github.com/szchengmi/kaggle-ai-series.git
```

### Step 2: 下载模型（首次，约15-30分钟）

```python
%cd /kaggle/working/kaggle-ai-series/scripts
!python download_models.py
```

下载完成后，Kaggle页面 → Output → **Save as Dataset**，名称填 `kaggle-ai-series-models`

### Step 3: 运行流水线

```python
%cd /kaggle/working/kaggle-ai-series/scripts
!python kaggle_pipeline.py
```

如果已挂载Dataset，模型自动复用。

## 设置Kaggle Secrets（可选）

Notebook 左侧 **Add-ons → Secrets**：
- `GOOGLE_API_KEY` — [获取](https://aistudio.google.com/apikey)（不设置则用本地Qwen）
- `HF_TOKEN` — [获取](https://huggingface.co/settings/tokens)（可选）

## 项目结构

```
scripts/
├── download_models.py          # 🔽 模型下载（首次运行）
├── kaggle_pipeline.py          # 🔥 端到端主流程
├── config/config.env           # 参数配置
└── requirements.txt            # Python依赖
```

## 模型列表

| 模型 | 用途 | 大小 |
|------|------|------|
| runwayml/stable-diffusion-v1-5 | SD 1.5 画面生成 | ~2.43GB |
| stabilityai/sd-vae-ft-mse | VAE改进版 | ~334MB |
| guoyww/animatediff-motion-adapter-v1-5-2 | AnimateDiff视频 | ~301MB |
| Qwen/Qwen2.5-3B-Instruct | 本地LLM备用 | ~6.44GB |

总计约 **9.5GB**，Kaggle Dataset上限20GB，够用。

## 耗时（T4 GPU）

| 步骤 | 时间 |
|------|------|
| 模型下载 | ~15-30分钟（一次性） |
| 剧本 | ~30秒 |
| 画面 | ~10-20分钟 |
| 视频 | ~20-40分钟 |
| 配音+剪辑 | ~10分钟 |
