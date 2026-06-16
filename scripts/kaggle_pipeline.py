"""
Kaggle AI短剧自动生成 - 端到端流水线
===================================
在Kaggle Notebook中运行此脚本，自动完成：
  1. 剧本生成 (Gemini API)
  2. 分镜生成 (结构化JSON)
  3. 画面生成 (SD 1.5)
  4. 视频生成 (AnimateDiff-Lightning)
  5. 配音生成 (ChatTTS / edge-tts)
  6. 剪辑合成 (FFmpeg)

兼容Kaggle预装的 google-genai 旧版本库。
"""

import os
import sys
import json
import time
import shutil
import subprocess
import torch

# ============================================================
# Kaggle Secrets 读取
# ============================================================

def get_kaggle_secret(key_name):
    """从Kaggle Secrets读取密钥（兼容kaggle_secrets库和环境变量）"""
    # 方式1: kaggle_secrets 库
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        val = client.get_secret(key_name)
        if val:
            return val
    except ImportError:
        pass
    except Exception:
        pass

    # 方式2: 环境变量
    val = os.environ.get(key_name, "")
    if val:
        return val

    return ""


# ============================================================
# 配置
# ============================================================

GOOGLE_API_KEY = get_kaggle_secret("GOOGLE_API_KEY")
HF_TOKEN = get_kaggle_secret("HF_TOKEN")

# Kaggle Secrets 特殊处理：Kaggle会把Secrets赋值给 secret_value_0, secret_value_1 等变量
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = secret_value_0  # noqa: F821
    except NameError:
        pass
if not HF_TOKEN:
    try:
        HF_TOKEN = secret_value_1  # noqa: F821
    except NameError:
        pass

EPISODE_NUM = int(os.environ.get("EPISODE_NUM", "1"))
GENRE = os.environ.get("GENRE", "urban_romance")
NUM_SCENES = int(os.environ.get("NUM_SCENES", "6"))
SHOTS_PER_SCENE = int(os.environ.get("SHOTS_PER_SCENE", "3"))

QUALITY_MODE = os.environ.get("QUALITY_MODE", "fast")
QUALITY_PRESETS = {
    "fast": {"steps": 15, "resolution": 512, "fps": 8},
    "balanced": {"steps": 20, "resolution": 512, "fps": 8},
    "quality": {"steps": 30, "resolution": 768, "fps": 12},
}
PRESET = QUALITY_PRESETS.get(QUALITY_MODE, QUALITY_PRESETS["fast"])

IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", str(PRESET["steps"])))
IMAGE_GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "7.5"))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", str(PRESET["fps"])))
VIDEO_RESOLUTION = int(os.environ.get("VIDEO_RESOLUTION", str(PRESET["resolution"])))
AUDIO_SAMPLE_RATE = 24000

if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ============================================================
# 目录
# ============================================================

BASE_DIR = "/kaggle/working/ai-series"
# 模型缓存目录 — 递归搜索 /kaggle/input/ 下所有 models/ 目录
MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"  # fallback
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if "models" in _dirs:
        _candidate = os.path.join(_root, "models")
        # 检查是否包含我们需要的模型子目录
        if os.path.isdir(os.path.join(_candidate, "stable-diffusion-v1-5")):
            MODEL_CACHE_DIR = _candidate
            break

# HuggingFace模型缓存
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
# 模型权重从本地加载，config允许下载（HF Hub需要联网获取pipeline配置）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # transformers离线（Qwen/tokenizer）
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

log(f"[OK] MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")

# ============================================================
# 检测Dataset中的模型文件
# ============================================================
REQUIRED_MODELS = {
    "SD 1.5": {
        "path": f"{MODEL_CACHE_DIR}/stable-diffusion-v1-5",
        "files": ["v1-5-pruned-emaonly.safetensors", "model_index.json"],
        "min_size_mb": 2000,
    },
    "AnimateDiff": {
        "path": f"{MODEL_CACHE_DIR}/animatediff",
        "files": ["diffusion_pytorch_model.safetensors", "config.json"],
        "min_size_mb": 200,
    },
    "Qwen2.5-3B": {
        "path": f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct",
        "files": ["config.json", "tokenizer.json", "model-00001-of-00002.safetensors"],
        "min_size_mb": 3000,
    },
}

log("=" * 50)
log("模型检测")
log("=" * 50)
all_ok = True
for name, spec in REQUIRED_MODELS.items():
    mp = spec["path"]
    if not os.path.isdir(mp):
        log(f"  ❌ {name}: 目录不存在 {mp}")
        all_ok = False
        continue
    missing = [f for f in spec["files"] if not os.path.isfile(f"{mp}/{f}")]
    if missing:
        log(f"  ❌ {name}: 缺少文件 {missing}")
        all_ok = False
        continue
    size_mb = sum(os.path.getsize(f"{mp}/{f}") for f in os.listdir(mp) if os.path.isfile(f"{mp}/{f}")) / 1e6
    if size_mb < spec["min_size_mb"]:
        log(f"  ⚠️  {name}: {size_mb:.0f}MB (可能不完整, 期望>{spec['min_size_mb']}MB)")
        all_ok = False
    else:
        log(f"  ✅ {name}: {size_mb:.0f}MB")

if not all_ok:
    log("")
    log("⚠️  模型不完整！Dataset可能缺少模型文件。")
    log("请确保Dataset包含以下目录:")
    log(f"  {MODEL_CACHE_DIR}/stable-diffusion-v1-5/")
    log(f"  {MODEL_CACHE_DIR}/animatediff/")
    log(f"  {MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct/")
    log("")
    log("或者运行下载脚本: python download_models.py")
else:
    log("全部模型就绪 ✅")
log("=" * 50)

# 代理支持（解决Gemini 403 / HF下载慢）
PROXY_URL = get_kaggle_secret("PROXY_URL")
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    print(f"[OK] 代理: {PROXY_URL}")
else:
    print("[INFO] 未设置代理 (Kaggle Secrets 名称: PROXY_URL)")

def get_dirs(episode_num=EPISODE_NUM):
    ep_dir = f"{BASE_DIR}/episode_{episode_num:02d}"
    return {
        "base": BASE_DIR, "episode": ep_dir,
        "storyboard": f"{ep_dir}/storyboards",
        "images": f"{ep_dir}/images",
        "videos": f"{ep_dir}/videos",
        "audio": f"{ep_dir}/audio",
        "final": f"{ep_dir}/final",
        "models": MODEL_CACHE_DIR,
        "logs": f"{ep_dir}/logs",
        "output": "/kaggle/working/output",
    }

def setup_dirs(episode_num=EPISODE_NUM):
    dirs = get_dirs(episode_num)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ============================================================
# 角色/场景设定
# ============================================================

CHARACTER_PROMPTS = {
    "xiaoming": {
        "base_prompt": "1boy, young Chinese man, short black hair, wearing glasses, wearing dark hoodie, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 42
    },
    "xiaoli": {
        "base_prompt": "1girl, young Chinese woman, long black hair, wearing light-colored dress, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 123
    },
    "boss_wang": {
        "base_prompt": "1man, middle-aged Chinese man, square face, thick eyebrows, wearing business suit, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 456
    }
}

SCENE_PROMPTS = {
    "office": "modern office interior, floor-to-ceiling windows, minimalist design, warm tones, anime background",
    "cafe": "cozy cafe interior, wooden furniture, warm yellow lighting, anime background",
    "park": "city park, green trees, bench and fountain, anime background",
    "apartment": "cozy apartment, Nordic style, living room, anime background",
    "street": "city street at dusk, street lamps, anime background"
}

EMOTION_ENHANCE = {
    "happy": "smiling, bright expression", "sad": "sad expression, teary eyes",
    "angry": "angry expression, furrowed brows", "surprised": "surprised expression, wide eyes",
    "nervous": "nervous expression, sweating", "calm": "calm expression, relaxed",
    "determined": "determined expression, confident", "embarrassed": "embarrassed expression, blushing",
    "thoughtful": "thoughtful expression, contemplative"
}

SHOT_PARAMS = {
    "close_up": {"w": 768, "h": 768, "prefix": "close-up shot of"},
    "medium_shot": {"w": 768, "h": 512, "prefix": "medium shot of"},
    "wide_shot": {"w": 1024, "h": 576, "prefix": "wide shot of"},
    "extreme_close_up": {"w": 512, "h": 768, "prefix": "extreme close-up of"}
}

VOICE_PARAMS = {
    "xiaoming": {"speed": 0.9, "temp": 0.3, "top_p": 0.7, "top_k": 20},
    "xiaoli": {"speed": 1.1, "temp": 0.4, "top_p": 0.8, "top_k": 25},
    "boss_wang": {"speed": 0.85, "temp": 0.25, "top_p": 0.6, "top_k": 15},
    "narrator": {"speed": 1.0, "temp": 0.3, "top_p": 0.7, "top_k": 20}
}

EMOTION_SPEED = {
    "happy": 1.1, "sad": 0.85, "angry": 1.15, "surprised": 1.2,
    "nervous": 1.1, "calm": 1.0, "determined": 1.05,
    "embarrassed": 1.1, "thoughtful": 0.9
}


# ============================================================
# Step 1: 剧本生成 (Gemini API)
# 兼容 google-genai 新旧版本
# ============================================================

def _build_story_prompt():
    """构建剧本生成的prompt"""
    return f"""你是一个专业的中文短剧编剧。请为一部{GENRE}题材的AI短剧写第{EPISODE_NUM}集的完整剧本。

【角色设定】
- 小明(xiaoming): 28岁, 程序员, 内向但善良, 戴眼镜, 短发, 常穿深色卫衣
- 小丽(xiaoli): 26岁, 平面设计师, 活泼开朗, 长发, 穿浅色连衣裙
- 王总(boss_wang): 45岁, 公司总监, 严厉但公正, 西装革履

【可用场景】
- office: 现代办公室, 落地窗, 简约风格, 暖色调
- cafe: 温馨咖啡馆, 木质桌椅, 暖黄灯光
- park: 城市公园, 绿树成荫, 长椅和喷泉
- apartment: 温馨公寓, 北欧风格
- street: 城市街道, 傍晚, 路灯

【要求】
1. 时长3-5分钟（约800-1200字）
2. 包含{NUM_SCENES}个场景，每个场景{SHOTS_PER_SCENE}个镜头
3. 完整故事线：开头→发展→高潮→结尾（留悬念）
4. 对话口语化，符合角色性格
5. 包含场景描述、角色动作、对话、旁白
6. 结尾留悬念

输出纯JSON（不要markdown标记，不要```json```）：
{{"episode": {EPISODE_NUM}, "title": "标题", "duration_estimate": "3-5分钟",
"scenes": [{{"scene_id": "scene_1", "location": "office", "time_of_day": "morning",
"lighting": "自然光", "mood": "描述氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作描述", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}],
"characters_used": ["xiaoming", "xiaoli"], "next_episode_hook": "下集预告"}}"""


def _parse_script_response(text):
    """解析LLM返回的JSON剧本"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return json.loads(text)


def _generate_with_gemini(prompt):
    """用Gemini API生成（支持代理）"""
    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)
    # 代理通过环境变量 HTTPS_PROXY 自动生效（httpx库会读取）
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={
            "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192,
        }
    )
    return response.text


def _generate_with_local_llm(prompt):
    """用本地Qwen2.5-3B生成（Kaggle T4 GPU）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch as _torch

    # 优先从Dataset/本地目录加载
    local_path = f"{MODEL_CACHE_DIR}/Qwen2.5-3B-Instruct"
    if os.path.isdir(local_path) and os.path.isfile(f"{local_path}/config.json"):
        model_path = local_path
        log(f"从本地加载: {model_path}")
    else:
        model_path = "Qwen/Qwen2.5-3B-Instruct"
        log(f"从HF下载: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # 分片模型需要用safetensors.index.json找到所有分片
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        use_safetensors=True,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with _torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=4096, temperature=0.9, top_p=0.95,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)
    return response_text


def step1_generate_script():
    log("=" * 50)
    log("Step 1: 剧本生成 (本地LLM)")
    log("=" * 50)

    prompt = _build_story_prompt()
    text = None

    # 优先用本地LLM（无需API Key）
    try:
        text = _generate_with_local_llm(prompt)
        log("本地LLM 成功")
    except Exception as e:
        log(f"本地LLM 失败: {e}")

    # 降级：Gemini API
    if text is None and GOOGLE_API_KEY:
        log("尝试 Gemini API...")
        try:
            text = _generate_with_gemini(prompt)
            log("Gemini API 成功")
        except Exception as e:
            log(f"Gemini API 失败: {e}")

    # 最终降级：预置剧本
    if text is None:
        log("使用预置剧本")
        text = _get_fallback_script()

    script_data = _parse_script_response(text)

    dirs = get_dirs()
    save_json(script_data, f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json")

    log(f"剧本: {script_data.get('title')}")
    total_shots = sum(len(s.get("shots", [])) for s in script_data.get("scenes", []))
    log(f"场景: {len(script_data.get('scenes', []))} | 镜头: {total_shots}")
    return script_data


def _get_fallback_script():
    """预置兜底剧本（所有LLM都失败时使用）"""
    return json.dumps({
        "episode": EPISODE_NUM, "title": "第一集：初遇",
        "duration_estimate": "3-5分钟",
        "scenes": [
            {
                "scene_id": "scene_1", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "平静日常",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明在办公室敲代码", "character": "xiaoming",
                     "action": "专注地敲击键盘", "dialogue": "", "narration": "周一的早晨，办公室里只有键盘的声音。",
                     "emotion": "calm", "subtitle": "周一的早晨，办公室里只有键盘的声音。"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明表情特写", "character": "xiaoming",
                     "action": "微微皱眉", "dialogue": "这个需求又改了...", "narration": "",
                     "emotion": "thoughtful", "subtitle": "这个需求又改了..."},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小丽走进办公室", "character": "xiaoli",
                     "action": "推门进来，笑着打招呼", "dialogue": "早啊小明！今天天气真好！", "narration": "",
                     "emotion": "happy", "subtitle": "早啊小明！今天天气真好！"}
                ]
            },
            {
                "scene_id": "scene_2", "location": "cafe", "time_of_day": "afternoon",
                "lighting": "暖黄灯光", "mood": "温馨浪漫",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明和小丽在咖啡馆聊天", "character": "xiaoming",
                     "action": "端着咖啡杯，认真倾听", "dialogue": "你觉得这个设计方案怎么样？", "narration": "",
                     "emotion": "calm", "subtitle": "你觉得这个设计方案怎么样？"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小丽眼睛发亮", "character": "xiaoli",
                     "action": "眼睛发亮，兴奋地比划", "dialogue": "我觉得配色可以再大胆一些！", "narration": "",
                     "emotion": "happy", "subtitle": "我觉得配色可以再大胆一些！"},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "pan_right",
                     "duration_seconds": 3, "description": "两人相视而笑", "character": "xiaoming",
                     "action": "忍不住笑了", "dialogue": "你总是这么有想法。", "narration": "",
                     "emotion": "embarrassed", "subtitle": "你总是这么有想法。"}
                ]
            },
            {
                "scene_id": "scene_3", "location": "office", "time_of_day": "evening",
                "lighting": "夕阳余晖", "mood": "紧张",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "王总走进办公室", "character": "boss_wang",
                     "action": "严肃地推门进来", "dialogue": "小明，客户对方案很不满意！", "narration": "",
                     "emotion": "angry", "subtitle": "小明，客户对方案很不满意！"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明紧张的表情", "character": "xiaoming",
                     "action": "紧张地站起来", "dialogue": "什么？我明明按需求做的...", "narration": "",
                     "emotion": "nervous", "subtitle": "什么？我明明按需求做的..."},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "三人对峙", "character": "boss_wang",
                     "action": "将文件摔在桌上", "dialogue": "需求已经变了，你不知道吗？明天早上之前改好！", "narration": "",
                     "emotion": "angry", "subtitle": "需求已经变了，你不知道吗？明天早上之前改好！"}
                ]
            },
            {
                "scene_id": "scene_4", "location": "apartment", "time_of_day": "night",
                "lighting": "台灯光", "mood": "温馨感人",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明在公寓加班", "character": "xiaoming",
                     "action": "疲惫地盯着屏幕", "dialogue": "", "narration": "夜深了，小明还在改方案。",
                     "emotion": "sad", "subtitle": "夜深了，小明还在改方案。"},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小丽端着夜宵进来", "character": "xiaoli",
                     "action": "轻轻推门，端着夜宵", "dialogue": "还没休息？我给你带了宵夜。", "narration": "",
                     "emotion": "calm", "subtitle": "还没休息？我给你带了宵夜。"},
                    {"shot_id": "shot_3", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明感动地看着小丽", "character": "xiaoming",
                     "action": "感动地看着小丽", "dialogue": "谢谢你，小丽。有你在真好。", "narration": "",
                     "emotion": "happy", "subtitle": "谢谢你，小丽。有你在真好。"}
                ]
            },
            {
                "scene_id": "scene_5", "location": "park", "time_of_day": "morning",
                "lighting": "阳光明媚", "mood": "充满希望",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "wide_shot", "camera_movement": "pan_left",
                     "duration_seconds": 3, "description": "公园里晨跑", "character": "xiaoming",
                     "action": "在公园晨跑", "dialogue": "", "narration": "改完方案的第二天，小明决定出门透透气。",
                     "emotion": "calm", "subtitle": "改完方案的第二天，小明决定出门透透气。"},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "偶遇小丽", "character": "xiaoli",
                     "action": "惊喜地挥手", "dialogue": "小明！好巧啊！", "narration": "",
                     "emotion": "surprised", "subtitle": "小明！好巧啊！"},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "两人并肩走在公园", "character": "xiaoming",
                     "action": "并肩散步，相视而笑", "dialogue": "小丽，昨晚的方案客户通过了！", "narration": "",
                     "emotion": "happy", "subtitle": "小丽，昨晚的方案客户通过了！"}
                ]
            },
            {
                "scene_id": "scene_6", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "紧张期待",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "王总宣布消息", "character": "boss_wang",
                     "action": "站在会议室前方", "dialogue": "告诉大家一个好消息——", "narration": "",
                     "emotion": "calm", "subtitle": "告诉大家一个好消息——"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明和小丽紧张对视", "character": "xiaoming",
                     "action": "紧张地握紧拳头", "dialogue": "", "narration": "",
                     "emotion": "nervous", "subtitle": ""},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "dolly_in",
                     "duration_seconds": 3, "description": "王总微笑", "character": "boss_wang",
                     "action": "露出罕见的微笑", "dialogue": "客户非常满意！小明、小丽，你们做到了！", "narration": "",
                     "emotion": "happy", "subtitle": "客户非常满意！小明、小丽，你们做到了！"}
                ]
            }
        ],
        "characters_used": ["xiaoming", "xiaoli", "boss_wang"],
        "next_episode_hook": "小明和小丽的项目获得了成功，但新的挑战正在等着他们..."
    }, ensure_ascii=False)


# ============================================================
# Step 2: 分镜生成
# ============================================================

def step2_generate_storyboard(script_data):
    log("=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)

    storyboard = {"episode": script_data.get("episode", 1), "title": script_data.get("title", ""), "characters": {}, "scenes": []}
    characters_used = set()

    for scene in script_data.get("scenes", []):
        scene_data = {
            "scene_id": scene["scene_id"], "location": scene["location"],
            "time_of_day": scene["time_of_day"], "lighting": scene["lighting"], "mood": scene["mood"], "shots": []
        }
        for shot in scene.get("shots", []):
            char = shot.get("character", "none")
            shot_type = shot.get("shot_type", "medium_shot")
            emotion = shot.get("emotion", "calm")
            if char != "none": characters_used.add(char)
            params = SHOT_PARAMS.get(shot_type, SHOT_PARAMS["medium_shot"])

            pp = [params["prefix"]]
            if char != "none" and char in CHARACTER_PROMPTS:
                pp.append(CHARACTER_PROMPTS[char]["base_prompt"])
            pp.append(shot.get("action", ""))
            if emotion in EMOTION_ENHANCE: pp.append(EMOTION_ENHANCE[emotion])
            sp = SCENE_PROMPTS.get(scene["location"], "")
            if sp: pp.append(sp)
            pp.append(f"{scene['time_of_day']}, {scene['lighting']}")
            pp.append(shot.get("description", ""))
            pp.append("masterpiece, best quality, detailed, anime style")

            neg = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, cropped, worst quality, low quality, jpeg artifacts, blurry"
            if char != "none" and char in CHARACTER_PROMPTS:
                neg += ", " + CHARACTER_PROMPTS[char]["negative_prompt"]

            scene_data["shots"].append({
                "shot_id": shot["shot_id"], "shot_type": shot_type,
                "camera_movement": shot.get("camera_movement", "static"),
                "duration_seconds": shot.get("duration_seconds", 3),
                "width": params["w"], "height": params["h"],
                "prompt": ", ".join([p for p in pp if p]),
                "negative_prompt": neg,
                "seed": CHARACTER_PROMPTS.get(char, {}).get("seed", -1),
                "character": char, "dialogue": shot.get("dialogue", ""),
                "narration": shot.get("narration", ""), "subtitle": shot.get("subtitle", ""),
                "description": shot.get("description", ""), "action": shot.get("action", ""),
                "emotion": emotion, "steps": IMAGE_STEPS, "guidance": IMAGE_GUIDANCE
            })
        storyboard["scenes"].append(scene_data)

    for cid in characters_used:
        storyboard["characters"][cid] = CHARACTER_PROMPTS[cid]

    dirs = get_dirs()
    save_json(storyboard, f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"分镜完成: {len(storyboard['scenes'])}场景 | {total}镜头")
    return storyboard


# ============================================================
# Step 3: 画面生成 (SD 1.5)
# ============================================================

def step3_generate_images(storyboard):
    log("=" * 50)
    log("Step 3: 画面生成 (SD 1.5)")
    log("=" * 50)

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    dirs = get_dirs()
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

    mp = f"{dirs['models']}/stable-diffusion-v1-5"
    if not os.path.exists(mp) or not os.listdir(mp):
        mp = "runwayml/stable-diffusion-v1-5"

    sd_file = f"{mp}/v1-5-pruned-emaonly.safetensors"
    if not os.path.isfile(sd_file):
        sd_file = "runwayml/stable-diffusion-v1-5"
        log(f"加载SD (HF): {sd_file}")
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_file, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        log(f"加载SD (单文件): {sd_file}")
        # 缓存config到working目录（Dataset只读）
        sd_cache = f"{BASE_DIR}/cache/stable-diffusion-v1-5"
        os.makedirs(sd_cache, exist_ok=True)
        os.environ["HF_HOME"] = sd_cache
        pipe = StableDiffusionPipeline.from_single_file(
            sd_file, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
            cache_dir=sd_cache,
        )

    # VAE: from_single_file已经包含vae，不需要额外加载
    # 只在vae为None时才尝试加载外部VAE
    if pipe.vae is None:
        try:
            from diffusers import AutoencoderKL
            vp = f"{dirs['models']}/vae-ft-mse"
            if not os.path.exists(vp): vp = "stabilityai/sd-vae-ft-mse"
            pipe.vae = AutoencoderKL.from_pretrained(vp, torch_dtype=DTYPE)
        except Exception as e:
            log(f"VAE: {e}")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if has_gpu:
        try: pipe.enable_attention_slicing()
        except: pass
        try: pipe.enable_vae_slicing()
        except: pass
        pipe.to(device)
    else:
        pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("xformers ✓")
    except:
        pass

    log(f"生成 {total} 张...")
    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['images']}/ep{ep:02d}_{scene['scene_id']}_{sid}.png"
            if os.path.exists(out): continue

            w = max((shot["width"] // 8) * 8, 512)
            h = max((shot["height"] // 8) * 8, 512)
            gen = None
            if shot.get("seed", -1) > 0:
                gen = torch.Generator(device).manual_seed(shot["seed"])

            try:
                result = pipe(prompt=shot["prompt"], negative_prompt=shot.get("negative_prompt", ""),
                             width=w, height=h,
                             num_inference_steps=shot.get("steps", IMAGE_STEPS),
                             guidance_scale=shot.get("guidance", IMAGE_GUIDANCE), generator=gen)
                result.images[0].save(out)
                log(f"  [{count}/{total}] {sid} ({w}x{h}) ✓")
            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_image(shot, out)

            if has_gpu and count % 5 == 0: torch.cuda.empty_cache()

    log("画面生成完成")


def _placeholder_images(storyboard):
    dirs = get_dirs()
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            ep = storyboard.get("episode", 1)
            out = f"{dirs['images']}/ep{ep:02d}_{scene['scene_id']}_{shot['shot_id']}.png"
            if not os.path.exists(out):
                _save_placeholder_image(shot, out)


def _save_placeholder_image(shot, output_path):
    from PIL import Image, ImageDraw
    w = max((shot.get("width", 768) // 8) * 8, 512)
    h = max((shot.get("height", 768) // 8) * 8, 512)
    img = Image.new('RGB', (w, h), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, w-10, h-10], outline=(100, 100, 200), width=2)
    draw.text((30, 40), f"[PLACEHOLDER] {shot.get('shot_id', '')}", fill=(200, 200, 255))
    draw.text((30, 80), f"Char: {shot.get('character', '')}", fill=(200, 255, 200))
    draw.text((30, 120), f"Desc: {shot.get('description', '')[:50]}", fill=(255, 255, 200))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


# ============================================================
# Step 4: 视频生成 (AnimateDiff-Lightning)
# ============================================================

def step4_generate_videos(storyboard):
    log("=" * 50)
    log("Step 4: 视频生成 (AnimateDiff-Lightning)")
    log("=" * 50)

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    dirs = get_dirs()
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    from diffusers import StableDiffusionPipeline, MotionAdapter, EulerAncestralDiscreteScheduler

    mp = f"{dirs['models']}/stable-diffusion-v1-5"
    if not os.path.exists(mp) or not os.listdir(mp): mp = "runwayml/stable-diffusion-v1-5"
    motp = f"{dirs['models']}/animatediff"
    if not os.path.exists(motp) or not os.listdir(motp): motp = "guoyww/animatediff-motion-adapter-v1-5-2"

    log("加载AnimateDiff...")
    adapter = MotionAdapter.from_pretrained(motp, torch_dtype=DTYPE, local_files_only=True, cache_dir=motp)
    sd_file = f"{mp}/v1-5-pruned-emaonly.safetensors"
    if os.path.isfile(sd_file):
        sd_cache = f"{BASE_DIR}/cache/stable-diffusion-v1-5"
        os.makedirs(sd_cache, exist_ok=True)
        pipe = StableDiffusionPipeline.from_single_file(
            sd_file, motion_adapter=adapter, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
            cache_dir=sd_cache,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            mp, motion_adapter=adapter, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
        )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear", steps_offset=1)
    if has_gpu:
        try: pipe.enable_attention_slicing()
        except: pass
        try: pipe.enable_vae_slicing()
        except: pass
        pipe.to(device)
    else:
        pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass

    motion_hints = {
        "static": "subtle motion, minimal movement", "pan_left": "camera panning left",
        "pan_right": "camera panning right", "dolly_in": "camera zooming in slowly", "dolly_out": "camera zooming out slowly"
    }

    log(f"生成 {total} 个视频...")
    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            if os.path.exists(out): continue

            dur = shot.get("duration_seconds", 3)
            nf = min(int(dur * VIDEO_FPS), 32)
            cm = shot.get("camera_movement", "static")
            enhanced = f"{shot['prompt']}, {motion_hints.get(cm, 'subtle motion')}, smooth animation"
            gen = None
            if shot.get("seed", -1) > 0: gen = torch.Generator(device).manual_seed(shot["seed"])

            try:
                result = pipe(prompt=enhanced, negative_prompt=shot.get("negative_prompt", ""),
                             width=VIDEO_RESOLUTION, height=VIDEO_RESOLUTION,
                             num_frames=nf, num_inference_steps=15, guidance_scale=7.5, generator=gen)
                # 兼容不同diffusers版本
                if hasattr(result, 'frames'):
                    frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames
                elif hasattr(result, 'images'):
                    frames = result.images
                else:
                    raise ValueError(f"未知的pipeline输出类型: {type(result)}")
                fd = out + "_frames"
                os.makedirs(fd, exist_ok=True)
                for i, f in enumerate(frames): f.save(f"{fd}/frame_{i:04d}.png")
                run_cmd(f'ffmpeg -y -framerate {VIDEO_FPS} -i "{fd}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{out}" 2>/dev/null')
                shutil.rmtree(fd, ignore_errors=True)
                log(f"  [{count}/{total}] {sid} ({nf}f) ✓")
            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_video(shot, out, nf)

            if has_gpu and count % 3 == 0: torch.cuda.empty_cache()

    log("视频生成完成")


def _placeholder_videos(storyboard):
    dirs = get_dirs()
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{shot['shot_id']}.mp4"
            if not os.path.exists(out):
                _save_placeholder_video(shot, out, min(int(shot.get("duration_seconds", 3) * VIDEO_FPS), 32))


def _save_placeholder_video(shot, output_path, num_frames):
    from PIL import Image, ImageDraw
    res = VIDEO_RESOLUTION
    frames = []
    for i in range(num_frames):
        img = Image.new('RGB', (res, res), (20, 20, 40))
        draw = ImageDraw.Draw(img)
        offset = int(i * 3 % 50)
        draw.rectangle([5, 5, res-5, res-5], outline=(100, 100, 200), width=2)
        draw.text((20, 30 + offset % 20), "[VIDEO]", fill=(200, 200, 255))
        draw.text((20, 70 + offset % 20), shot.get("shot_id", ""), fill=(200, 255, 200))
        frames.append(img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gif_path = output_path.replace(".mp4", ".gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / VIDEO_FPS), loop=0)
    run_cmd(f'ffmpeg -y -framerate {VIDEO_FPS} -i "{gif_path}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "{output_path}" 2>/dev/null')


# ============================================================
# Step 5: 配音生成 (ChatTTS / edge-tts)
# ============================================================

def step5_generate_audio(storyboard):
    log("=" * 50)
    log("Step 5: 配音生成 (ChatTTS)")
    log("=" * 50)

    dirs = get_dirs()
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    chat = None
    try:
        import ChatTTS
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        log("ChatTTS ✓")
    except Exception as e:
        log(f"ChatTTS: {e}")

    edge_ok = False
    try:
        import edge_tts
        edge_ok = True
    except: pass

    EDGE_V = {"xiaoming": "zh-CN-YunxiNeural", "xiaoli": "zh-CN-XiaoxiaoNeural", "boss_wang": "zh-CN-YunjianNeural", "narrator": "zh-CN-YunxiNeural"}

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['audio']}/ep{ep:02d}_{scene['scene_id']}_{sid}.wav"
            if os.path.exists(out): continue

            char = shot.get("character", "narrator")
            text = shot.get("dialogue") or shot.get("narration") or ""
            emotion = shot.get("emotion", "calm")
            dur = shot.get("duration_seconds", 3)

            if not text:
                _save_silence(out, float(dur))
                continue

            ok = False
            if chat is not None:
                try:
                    wavs = chat.infer([text])
                    if wavs and len(wavs) > 0:
                        import torchaudio
                        audio = wavs[0]
                        if isinstance(audio, torch.Tensor):
                            torchaudio.save(out, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                        ok = True
                except: pass

            if not ok and edge_ok:
                try:
                    import asyncio
                    vn = EDGE_V.get(char, "zh-CN-YunxiNeural")
                    async def _t():
                        c = edge_tts.Communicate(text, vn)
                        await c.save(out)
                    asyncio.run(_t())
                    ok = True
                except: pass

            if not ok: _save_silence(out, float(dur))
            log(f"  [{count}/{total}] {sid} ({char}) {'✓' if ok else '静音'}")

    log("配音生成完成")


def _save_silence(output_path, duration):
    try:
        import wave, struct
        sr = AUDIO_SAMPLE_RATE
        n = int(sr * duration)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            for _ in range(n): w.writeframes(struct.pack('<h', 0))
    except:
        run_cmd(f'ffmpeg -y -f lavfi -i "anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono" -t {duration} -acodec pcm_s16le "{output_path}" 2>/dev/null')


# ============================================================
# Step 6: 剪辑合成 (FFmpeg)
# ============================================================

def step6_compose(storyboard):
    log("=" * 50)
    log("Step 6: 剪辑合成 (FFmpeg)")
    log("=" * 50)

    dirs = get_dirs()
    ep = storyboard.get("episode", 1)

    srt = f"{dirs['final']}/ep{ep:02d}.srt"
    total_dur = _make_srt(storyboard, srt)

    vids = []
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            vp = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            gp = vp.replace(".mp4", ".gif")
            if os.path.exists(vp): vids.append(vp)
            elif os.path.exists(gp):
                run_cmd(f'ffmpeg -y -i "{gp}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "{vp}" 2>/dev/null')
                if os.path.exists(vp): vids.append(vp)

    auds = []
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            ap = f"{dirs['audio']}/ep{ep:02d}_{scene['scene_id']}_{sid}.wav"
            if os.path.exists(ap): auds.append(ap)

    log(f"视频: {len(vids)} | 音频: {len(auds)}")
    if not vids:
        log("[ERROR] 没有视频"); return

    cv = f"{dirs['final']}/_video.mp4"
    ca = f"{dirs['final']}/_audio.wav"
    _concat_files(vids, cv, "video")
    if auds: _concat_files(auds, ca, "audio")
    else: _save_silence(ca, total_dur)

    final = f"{dirs['final']}/episode_{ep:02d}_final.mp4"
    cmd = (
        f'ffmpeg -y -i "{cv}" -i "{ca}" '
        f'-vf "subtitles=\'{srt}\':force_style=\'FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2\'" '
        f'-c:v libx264 -crf 20 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -ar 44100 -ac 2 '
        f'-shortest -movflags +faststart "{final}"'
    )
    run_cmd(cmd, timeout=300)

    if os.path.exists(final):
        log(f"最终: {final} ({os.path.getsize(final)/1e6:.1f}MB)")
    else:
        cmd2 = (f'ffmpeg -y -i "{cv}" -i "{ca}" -c:v libx264 -crf 20 -pix_fmt yuv420p '
                f'-c:a aac -b:a 128k -ar 44100 -ac 2 -shortest -movflags +faststart "{final}"')
        run_cmd(cmd2, timeout=300)
        if os.path.exists(final): log(f"最终(无硬字幕): {final} ({os.path.getsize(final)/1e6:.1f}MB)")
        else: log("[FAIL] 合成失败")

    for f in [cv, ca]:
        if os.path.exists(f): os.remove(f)
    log("剪辑合成完成")


def _make_srt(storyboard, path):
    lines, idx, t = [], 1, 0.0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            dur = shot.get("duration_seconds", 3)
            text = shot.get("subtitle") or shot.get("dialogue") or ""
            if text:
                lines.extend([str(idx), f"{seconds_to_srt_time(t)} --> {seconds_to_srt_time(t+dur)}", text, ""])
                idx += 1
            t += dur
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    log(f"SRT: {idx-1}条 | {t:.1f}s")
    return t


def _concat_files(file_list, output, mtype):
    lf = output + ".list"
    with open(lf, "w") as f:
        for p in file_list: f.write(f"file '{p}'\n")
    if mtype == "video":
        cmd = f'ffmpeg -y -f concat -safe 0 -i "{lf}" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "{output}"'
    else:
        cmd = f'ffmpeg -y -f concat -safe 0 -i "{lf}" -acodec pcm_s16le -ar {AUDIO_SAMPLE_RATE} -ac 1 "{output}"'
    run_cmd(cmd, timeout=300)
    if os.path.exists(lf): os.remove(lf)


# ============================================================
# 主流程
# ============================================================

def main():
    log("╔══════════════════════════════════════════╗")
    log("║   AI短剧自动生成 - 端到端流水线            ║")
    log("╚══════════════════════════════════════════╝")
    log(f"集数: {EPISODE_NUM} | 题材: {GENRE} | 质量: {QUALITY_MODE}")
    log(f"步数: {IMAGE_STEPS} | 分辨率: {VIDEO_RESOLUTION} | FPS: {VIDEO_FPS}")

    t0 = time.time()
    setup_dirs()

    # 依赖
    _install_if_missing("diffusers", "diffusers transformers accelerate safetensors")
    _install_if_missing("ChatTTS", "ChatTTS soundfile edge-tts moviepy")

    global DTYPE
    has_gpu = torch.cuda.is_available()
    DTYPE = torch.float16 if has_gpu else torch.float32
    if has_gpu:
        log(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    else:
        log("模式: CPU (float32)")

    script = step1_generate_script()
    storyboard = step2_generate_storyboard(script)
    step3_generate_images(storyboard)
    step4_generate_videos(storyboard)
    step5_generate_audio(storyboard)
    step6_compose(storyboard)

    elapsed = (time.time() - t0) / 60
    log(f"\n{'=' * 50}")
    log(f"全部完成! 耗时: {elapsed:.1f} 分钟")

    # 复制最终视频和SRT到output目录（持久化）
    final_video = f"{get_dirs()['final']}/episode_{EPISODE_NUM:02d}_final.mp4"
    final_srt = f"{get_dirs()['final']}/ep{EPISODE_NUM:02d}.srt"
    output_dir = get_dirs()['output']
    if os.path.exists(final_video):
        shutil.copy2(final_video, f"{output_dir}/episode_{EPISODE_NUM:02d}_final.mp4")
        log(f"视频已复制到: {output_dir}/episode_{EPISODE_NUM:02d}_final.mp4")
    if os.path.exists(final_srt):
        shutil.copy2(final_srt, f"{output_dir}/ep{EPISODE_NUM:02d}.srt")

    log(f"模型缓存: {MODEL_CACHE_DIR}")
    log(f"输出目录: {output_dir}")
    log(f"{'=' * 50}")


def _install_if_missing(module_name, pip_packages):
    try:
        __import__(module_name)
    except ImportError:
        log(f"安装 {pip_packages}...")
        run_cmd(f"pip install -q {pip_packages}", timeout=120)
        try:
            run_cmd("pip install -q xformers", timeout=60)
        except: pass


if __name__ == "__main__":
    main()
