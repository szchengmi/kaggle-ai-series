# Kaggle AI短剧自动生成 - 端到端流水线

在Kaggle Notebook中运行，自动完成AI短剧的全流程生成：

1. **剧本生成** - Gemini API / 本地Qwen2.5-3B / 预置剧本（三级降级）
2. **分镜生成** - 结构化JSON
3. **画面生成** - Stable Diffusion 1.5
4. **视频生成** - AnimateDiff-Lightning
5. **配音生成** - ChatTTS / edge-tts
6. **剪辑合成** - FFmpeg

## Kaggle Notebook 使用（2步）

### Step 1: 下载模型（首次运行）

```python
!git clone https://github.com/szchengmi/kaggle-ai-series.git
%cd /kaggle/working/kaggle-ai-series/scripts
!python download_models.py
```

下载完成后，在Kaggle Output目录 **Save as Dataset**（名称：`kaggle-ai-series-models`）。

### Step 2: 运行流水线

```python
!git clone https://github.com/szchengmi/kaggle-ai-series.git
%cd /kaggle/working/kaggle-ai-series/scripts
!python kaggle_pipeline.py
```

如果已保存Dataset，在Notebook中 **Add Data → kaggle-ai-series-models** 挂载，模型会自动加载，无需重复下载。

## 设置Kaggle Secrets（可选）

在 Notebook 左侧 **Add-ons → Secrets** 中可以添加：
- `GOOGLE_API_KEY` — 从 [Google AI Studio](https://aistudio.google.com/apikey) 获取（可选，不设置则用本地Qwen模型）
- `HF_TOKEN` — 从 [HuggingFace](https://huggingface.co/settings/tokens) 获取（可选）

## 项目结构

```
scripts/
├── download_models.py          # 🔽 模型下载脚本（首次运行）
├── kaggle_pipeline.py          # 🔥 端到端主流程
├── config/config.env           # 参数配置
└── requirements.txt            # Python依赖
```

## 模型列表（download_models.py）

| 模型 | 用途 | 大小 |
|------|------|------|
| runwayml/stable-diffusion-v1-5 | SD 1.5 画面生成 | ~2.43GB |
| stabilityai/sd-vae-ft-mse | VAE改进版 | ~334MB |
| guoyww/animatediff-motion-adapter-v1-5-2 | AnimateDiff视频 | ~301MB |
| Qwen/Qwen2.5-3B-Instruct | 本地LLM备用 | ~6.44GB |

## 各步骤耗时（T4 GPU）

| 步骤 | 工具 | 时间 |
|------|------|------|
| 剧本 | Gemini API / Qwen本地 | ~30秒 / ~2分钟 |
| 分镜 | Python | ~5秒 |
| 画面 | SD 1.5 | ~10-20分钟 |
| 视频 | AnimateDiff-Lightning | ~20-40分钟 |
| 配音 | ChatTTS | ~5-10分钟 |
| 剪辑 | FFmpeg | ~1-2分钟 |
