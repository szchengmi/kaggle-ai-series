# Kaggle AI短剧自动生成 - 端到端流水线

在Kaggle Notebook中运行，自动完成AI短剧的全流程生成：

1. **剧本生成** - Gemini API / 本地Qwen2.5-3B / 预置剧本（三级降级）
2. **分镜生成** - 结构化JSON
3. **画面生成** - Stable Diffusion 1.5
4. **视频生成** - AnimateDiff-Lightning
5. **配音生成** - ChatTTS / edge-tts
6. **剪辑合成** - FFmpeg

## Kaggle Notebook 使用（3步）

### 1. 创建Notebook
- 访问 https://kaggle.com/notebooks → New Notebook
- 开启 **GPU (T4)** 和 **高RAM**

### 2. 设置Secrets（可选）
在 Notebook 左侧 **Add-ons → Secrets** 中可以添加：
- `GOOGLE_API_KEY` — 从 [Google AI Studio](https://aistudio.google.com/apikey) 获取（可选，不设置则用本地Qwen模型）
- `HF_TOKEN` — 从 [HuggingFace](https://huggingface.co/settings/tokens) 获取（可选）

> 注意：Kaggle的IP可能被Google封，Gemini API可能403。此时会自动降级为本地Qwen2.5-3B模型生成。

### 3. 第一个Cell克隆并运行
```python
!git clone https://github.com/szchengmi/kaggle-ai-series.git
%cd /kaggle/working/kaggle-ai-series/scripts
!python kaggle_pipeline.py
```

## 项目结构

```
scripts/
├── kaggle_pipeline.py          # 🔥 端到端主流程（Kaggle直接跑这个）
├── config/config.env           # 参数配置
└── requirements.txt            # Python依赖
```

## 参数修改

在 `kaggle_pipeline.py` 开头的 **配置区** 修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EPISODE_NUM` | 1 | 第几集 |
| `GENRE` | urban_romance | 题材 |
| `QUALITY_MODE` | fast | fast/balanced/quality |
| `IMAGE_STEPS` | 15 | SD推理步数 |
| `VIDEO_RESOLUTION` | 512 | 视频分辨率 |

## 各步骤耗时（T4 GPU）

| 步骤 | 工具 | 时间 |
|------|------|------|
| 剧本 | Gemini API / Qwen本地 | ~30秒 / ~2分钟 |
| 分镜 | Python | ~5秒 |
| 画面 | SD 1.5 | ~10-20分钟 |
| 视频 | AnimateDiff-Lightning | ~20-40分钟 |
| 配音 | ChatTTS | ~5-10分钟 |
| 剪辑 | FFmpeg | ~1-2分钟 |

## 剧本生成三级降级

1. **Gemini API** — 质量最高，需要API Key且Kaggle IP未被封
2. **本地Qwen2.5-3B** — 无需API Key，Kaggle T4可跑，质量中等
3. **预置剧本** — 兜底方案，保证流程能跑通
