# Kaggle AI短剧自动生成 - 使用说明

## 快速开始

### 1. 在Kaggle创建Notebook
- 访问 https://kaggle.com/notebooks
- 新建Notebook
- 开启 GPU (T4) 和 高RAM

### 2. 设置Secrets（API Keys）
在Notebook的 Add-ons → Secrets 中添加：
- `GOOGLE_API_KEY`: 你的Google AI Studio API Key
- `HF_TOKEN`: 你的HuggingFace Token（可选，用于下载模型）

### 3. 上传脚本
将 `kaggle_pipeline.py` 上传到Notebook，或在第一个Cell中粘贴全部代码

### 4. 运行
直接运行全部Cell即可

## 参数修改

在脚本开头的 **配置区** 修改：

```python
EPISODE_NUM = 1          # 第几集
GENRE = "urban_romance"  # 题材
QUALITY_MODE = "fast"    # fast/balanced/quality
IMAGE_STEPS = 20         # SD推理步数
VIDEO_RESOLUTION = 512   # 视频分辨率
```

## 输出目录结构

```
/kaggle/working/ai-series/
├── episode_01/
│   ├── storyboards/     # 剧本 + 分镜JSON
│   ├── images/          # SD生成的图片 (PNG)
│   ├── videos/          # AnimateDiff生成的视频 (MP4)
│   ├── audio/           # ChatTTS生成的配音 (WAV)
│   ├── final/           # 最终合成视频 (MP4 + SRT)
│   └── logs/
└── models/              # 模型缓存
```

## 各步骤说明

| 步骤 | 工具 | GPU需求 | 时间估算 |
|------|------|---------|----------|
| 1. 剧本 | Gemini API | 无 | ~30秒 |
| 2. 分镜 | Python JSON | 无 | ~5秒 |
| 3. 画面 | SD 1.5 | T4 GPU | ~10-20分钟 |
| 4. 视频 | AnimateDiff-Lightning | T4 GPU | ~20-40分钟 |
| 5. 配音 | ChatTTS / edge-tts | CPU | ~5-10分钟 |
| 6. 剪辑 | FFmpeg | CPU | ~1-2分钟 |

## 注意事项

- Kaggle无法直连huggingface.co，需要先下载模型到/kaggle/working/
- 如果OOM，降低 QUALITY_MODE 为 "fast"
- 最终视频在 `/kaggle/working/ai-series/episode_01/final/`
- 可以用 `from IPython.display import Video` 预览
