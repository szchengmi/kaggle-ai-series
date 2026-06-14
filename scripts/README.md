# Kaggle AI短剧自动生成 - 端到端流水线

在Kaggle Notebook中运行，自动完成AI短剧的全流程生成：

1. **剧本生成** - Gemini API
2. **分镜生成** - 结构化JSON
3. **画面生成** - Stable Diffusion 1.5
4. **视频生成** - AnimateDiff-Lightning
5. **配音生成** - ChatTTS / edge-tts
6. **剪辑合成** - FFmpeg

## 快速开始

### Kaggle Notebook 使用

1. 在 [Kaggle](https://kaggle.com/notebooks) 新建 Notebook
2. 开启 **GPU (T4)** 和 **高RAM**
3. 在 **Add-ons → Secrets** 中添加：
   - `GOOGLE_API_KEY` - 从 [Google AI Studio](https://aistudio.google.com/apikey) 获取
   - `HF_TOKEN` - 从 [HuggingFace](https://huggingface.co/settings/tokens) 获取（可选）
4. 在第一个 Cell 中克隆仓库：

```python
!git clone https://github.com/YOUR_USERNAME/kaggle-ai-series.git
%cd kaggle-ai-series/scripts
```

5. 直接运行 `kaggle_pipeline.py` 即可

### 本地测试

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
python scripts/kaggle_pipeline.py
```

## 项目结构

```
├── kaggle_pipeline.py          # 🔥 端到端主流程（Kaggle直接跑这个）
├── step1_generate_story.py     # Step 1: 剧本生成 (Gemini API)
├── step2_generate_storyboard.py # Step 2: 分镜生成
├── step3_generate_images.py    # Step 3: 画面生成 (SD 1.5)
├── step4_generate_videos.py    # Step 4: 视频生成 (AnimateDiff-Lightning)
├── step5_generate_audio.py     # Step 5: 配音生成 (ChatTTS)
├── step6_compose.py            # Step 6: 剪辑合成 (FFmpeg)
├── config/
│   └── config.env              # 参数配置
├── requirements.txt            # Python依赖
└── .gitignore
```

## 参数修改

在 `kaggle_pipeline.py` 开头的 **配置区** 或 `config/config.env` 中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EPISODE_NUM` | 1 | 第几集 |
| `GENRE` | urban_romance | 题材 |
| `QUALITY_MODE` | fast | fast/balanced/quality |
| `IMAGE_STEPS` | 20 | SD推理步数 |
| `VIDEO_RESOLUTION` | 512 | 视频分辨率 |

## 各步骤耗时（T4 GPU）

| 步骤 | 工具 | 时间 |
|------|------|------|
| 剧本 | Gemini API | ~30秒 |
| 分镜 | Python | ~5秒 |
| 画面 | SD 1.5 | ~10-20分钟 |
| 视频 | AnimateDiff-Lightning | ~20-40分钟 |
| 配音 | ChatTTS | ~5-10分钟 |
| 剪辑 | FFmpeg | ~1-2分钟 |
