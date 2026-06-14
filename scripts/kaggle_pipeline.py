#!/usr/bin/env python3
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

使用方法：
  1. 在Kaggle Notebook中新建Cell
  2. 粘贴全部代码（或 git clone 后运行）
  3. 设置环境变量 GOOGLE_API_KEY 和 HF_TOKEN
  4. 运行
"""

import os
import sys
import json
import time
import shutil
import subprocess
import torch

# 确保脚本目录在path里
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import *

# ============================================================
# 可选配置（覆盖默认值）
# ============================================================

GENRE = os.environ.get("GENRE", "urban_romance")
NUM_SCENES = int(os.environ.get("NUM_SCENES", "6"))
SHOTS_PER_SCENE = int(os.environ.get("SHOTS_PER_SCENE", "3"))


# ============================================================
# Step 1: 剧本生成 (Gemini API)
# ============================================================

def step1_generate_script():
    """生成剧本"""
    log("=" * 50)
    log("Step 1: 剧本生成 (Gemini API)")
    log("=" * 50)

    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

    prompt = f"""你是一个专业的中文短剧编剧。请为一部{GENRE}题材的AI短剧写第{EPISODE_NUM}集的完整剧本。

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

输出纯JSON（不要markdown标记）：
{{"episode": {EPISODE_NUM}, "title": "标题", "duration_estimate": "3-5分钟",
"scenes": [{{"scene_id": "scene_1", "location": "office", "time_of_day": "morning",
"lighting": "自然光", "mood": "描述氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作描述", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}],
"characters_used": ["xiaoming", "xiaoli"], "next_episode_hook": "下集预告"}}"""

    response = model.generate_content(prompt)

    # 解析JSON
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    script_data = json.loads(text)

    # 保存
    dirs = get_dirs()
    output_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"
    save_json(script_data, output_path)

    log(f"剧本: {script_data.get('title')}")
    total_shots = sum(len(s.get("shots", [])) for s in script_data.get("scenes", []))
    log(f"场景: {len(script_data.get('scenes', []))} | 镜头: {total_shots}")
    return script_data


# ============================================================
# Step 2: 分镜生成
# ============================================================

def step2_generate_storyboard(script_data):
    """生成分镜数据"""
    log("=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)

    storyboard = {
        "episode": script_data.get("episode", 1),
        "title": script_data.get("title", ""),
        "characters": {},
        "scenes": []
    }

    characters_used = set()

    for scene in script_data.get("scenes", []):
        scene_data = {
            "scene_id": scene["scene_id"],
            "location": scene["location"],
            "time_of_day": scene["time_of_day"],
            "lighting": scene["lighting"],
            "mood": scene["mood"],
            "shots": []
        }

        for shot in scene.get("shots", []):
            char = shot.get("character", "none")
            shot_type = shot.get("shot_type", "medium_shot")
            emotion = shot.get("emotion", "calm")

            if char != "none":
                characters_used.add(char)

            params = SHOT_PARAMS.get(shot_type, SHOT_PARAMS["medium_shot"])

            # 构建prompt
            prompt_parts = [params["prefix"]]
            if char != "none" and char in CHARACTER_PROMPTS:
                prompt_parts.append(CHARACTER_PROMPTS[char]["base_prompt"])
            prompt_parts.append(shot.get("action", ""))
            if emotion in EMOTION_ENHANCE:
                prompt_parts.append(EMOTION_ENHANCE[emotion])
            scene_prompt = SCENE_PROMPTS.get(scene["location"], "")
            if scene_prompt:
                prompt_parts.append(scene_prompt)
            prompt_parts.append(f"{scene['time_of_day']}, {scene['lighting']}")
            prompt_parts.append(shot.get("description", ""))
            prompt_parts.append("masterpiece, best quality, detailed, anime style")
            full_prompt = ", ".join([p for p in prompt_parts if p])

            negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, blurry"
            if char != "none" and char in CHARACTER_PROMPTS:
                negative += ", " + CHARACTER_PROMPTS[char]["negative_prompt"]

            shot_data = {
                "shot_id": shot["shot_id"],
                "shot_type": shot_type,
                "camera_movement": shot.get("camera_movement", "static"),
                "duration_seconds": shot.get("duration_seconds", 3),
                "width": params["w"],
                "height": params["h"],
                "prompt": full_prompt,
                "negative_prompt": negative,
                "seed": CHARACTER_PROMPTS.get(char, {}).get("seed", -1),
                "character": char,
                "dialogue": shot.get("dialogue", ""),
                "narration": shot.get("narration", ""),
                "subtitle": shot.get("subtitle", ""),
                "description": shot.get("description", ""),
                "action": shot.get("action", ""),
                "emotion": emotion,
                "steps": IMAGE_STEPS,
                "guidance": IMAGE_GUIDANCE
            }
            scene_data["shots"].append(shot_data)

        storyboard["scenes"].append(scene_data)

    for char_id in characters_used:
        storyboard["characters"][char_id] = CHARACTER_PROMPTS[char_id]

    dirs = get_dirs()
    output_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"
    save_json(storyboard, output_path)

    total_shots = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"分镜完成: {len(storyboard['scenes'])}场景 | {total_shots}镜头")
    return storyboard


# ============================================================
# Step 3: 画面生成 (SD 1.5)
# ============================================================

def step3_generate_images(storyboard):
    """生成所有镜头的图片"""
    log("=" * 50)
    log("Step 3: 画面生成 (SD 1.5)")
    log("=" * 50)

    has_gpu = torch.cuda.is_available()
    dirs = get_dirs()
    total_shots = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    if not has_gpu:
        log("[WARN] 无GPU，生成占位图")
        _placeholder_images(storyboard)
        return

    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

    model_path = f"{dirs['models']}/stable-diffusion-v1-5"
    if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
        model_path = "runwayml/stable-diffusion-v1-5"

    log(f"加载SD: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    )

    # VAE
    try:
        from diffusers import AutoencoderKL
        vae_path = f"{dirs['models']}/vae-ft-mse"
        if not os.path.exists(vae_path):
            vae_path = "stabilityai/sd-vae-ft-mse"
        pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    except Exception as e:
        log(f"VAE: {e}")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("xformers ✓")
    except:
        pass

    log(f"生成 {total_shots} 张图片...")
    count = 0

    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            shot_id = shot["shot_id"]
            output_path = f"{dirs['images']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{shot_id}.png"

            if os.path.exists(output_path):
                continue

            w = max((shot["width"] // 8) * 8, 512)
            h = max((shot["height"] // 8) * 8, 512)
            gen = None
            if shot.get("seed", -1) > 0:
                gen = torch.Generator("cuda").manual_seed(shot["seed"])

            try:
                result = pipe(
                    prompt=shot["prompt"],
                    negative_prompt=shot.get("negative_prompt", ""),
                    width=w, height=h,
                    num_inference_steps=shot.get("steps", IMAGE_STEPS),
                    guidance_scale=shot.get("guidance", IMAGE_GUIDANCE),
                    generator=gen
                )
                result.images[0].save(output_path)
                log(f"  [{count}/{total_shots}] {shot_id} ({w}x{h}) ✓")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  [{count}/{total_shots}] {shot_id} OOM!")
                _save_placeholder_image(shot, output_path)

            if count % 5 == 0:
                torch.cuda.empty_cache()

    log("画面生成完成")


def _placeholder_images(storyboard):
    """占位图"""
    dirs = get_dirs()
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            output_path = f"{dirs['images']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{shot['shot_id']}.png"
            if not os.path.exists(output_path):
                _save_placeholder_image(shot, output_path)


def _save_placeholder_image(shot, output_path):
    """保存占位图"""
    from PIL import Image, ImageDraw
    w = max((shot.get("width", 768) // 8) * 8, 512)
    h = max((shot.get("height", 768) // 8) * 8, 512)
    img = Image.new('RGB', (w, h), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, w-10, h-10], outline=(100, 100, 200), width=2)
    draw.text((30, 40), f"[PLACEHOLDER] {shot.get('shot_id', '')}", fill=(200, 200, 255))
    draw.text((30, 80), f"Char: {shot.get('character', '')}", fill=(200, 255, 200))
    desc = shot.get("description", "")[:50]
    draw.text((30, 120), f"Desc: {desc}", fill=(255, 255, 200))
    if shot.get("dialogue"):
        draw.text((30, 160), f"Dial: {shot['dialogue'][:30]}", fill=(255, 200, 200))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


# ============================================================
# Step 4: 视频生成 (AnimateDiff-Lightning)
# ============================================================

def step4_generate_videos(storyboard):
    """生成视频片段"""
    log("=" * 50)
    log("Step 4: 视频生成 (AnimateDiff-Lightning)")
    log("=" * 50)

    has_gpu = torch.cuda.is_available()
    dirs = get_dirs()
    total_shots = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    if not has_gpu:
        log("[WARN] 无GPU，生成占位视频")
        _placeholder_videos(storyboard)
        return

    from diffusers import StableDiffusionPipeline, MotionAdapter, EulerAncestralDiscreteScheduler

    model_path = f"{dirs['models']}/stable-diffusion-v1-5"
    if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
        model_path = "runwayml/stable-diffusion-v1-5"

    motion_path = f"{dirs['models']}/animatediff"
    if not os.path.exists(motion_path) or len(os.listdir(motion_path)) == 0:
        motion_path = "guoyww/animatediff-motion-adapter-v1-5-2"

    log("加载AnimateDiff...")
    adapter = MotionAdapter.from_pretrained(motion_path, torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, motion_adapter=adapter, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", steps_offset=1
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    motion_hints = {
        "static": "subtle motion, minimal movement",
        "pan_left": "camera panning left",
        "pan_right": "camera panning right",
        "dolly_in": "camera zooming in slowly",
        "dolly_out": "camera zooming out slowly"
    }

    log(f"生成 {total_shots} 个视频...")
    count = 0

    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            shot_id = shot["shot_id"]
            output_path = f"{dirs['videos']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{shot_id}.mp4"

            if os.path.exists(output_path):
                continue

            duration = shot.get("duration_seconds", 3)
            num_frames = min(int(duration * VIDEO_FPS), 32)
            camera_move = shot.get("camera_movement", "static")
            enhanced = f"{shot['prompt']}, {motion_hints.get(camera_move, 'subtle motion')}, smooth animation"

            gen = None
            if shot.get("seed", -1) > 0:
                gen = torch.Generator("cuda").manual_seed(shot["seed"])

            try:
                result = pipe(
                    prompt=enhanced,
                    negative_prompt=shot.get("negative_prompt", ""),
                    width=VIDEO_RESOLUTION, height=VIDEO_RESOLUTION,
                    num_frames=num_frames,
                    num_inference_steps=15,
                    guidance_scale=7.5,
                    generator=gen
                )
                frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames

                # 保存frames → ffmpeg
                frames_dir = output_path + "_frames"
                os.makedirs(frames_dir, exist_ok=True)
                for i, f in enumerate(frames):
                    f.save(f"{frames_dir}/frame_{i:04d}.png")

                run_cmd(
                    f'ffmpeg -y -framerate {VIDEO_FPS} -i "{frames_dir}/frame_%04d.png" '
                    f'-c:v libx264 -pix_fmt yuv420p -crf 23 '
                    f'-movflags +faststart "{output_path}" 2>/dev/null'
                )
                shutil.rmtree(frames_dir, ignore_errors=True)
                log(f"  [{count}/{total_shots}] {shot_id} ({num_frames}f) ✓")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  [{count}/{total_shots}] {shot_id} OOM!")
                _save_placeholder_video(shot, output_path, num_frames)

            if count % 3 == 0:
                torch.cuda.empty_cache()

    log("视频生成完成")


def _placeholder_videos(storyboard):
    dirs = get_dirs()
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            output_path = f"{dirs['videos']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{shot['shot_id']}.mp4"
            if not os.path.exists(output_path):
                duration = shot.get("duration_seconds", 3)
                _save_placeholder_video(shot, output_path, min(int(duration * VIDEO_FPS), 32))


def _save_placeholder_video(shot, output_path, num_frames):
    """保存占位视频"""
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
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / VIDEO_FPS), loop=0)
    run_cmd(f'ffmpeg -y -framerate {VIDEO_FPS} -i "{gif_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart '
            f'"{output_path}" 2>/dev/null')


# ============================================================
# Step 5: 配音生成 (ChatTTS + edge-tts备用)
# ============================================================

def step5_generate_audio(storyboard):
    """生成配音"""
    log("=" * 50)
    log("Step 5: 配音生成 (ChatTTS)")
    log("=" * 50)

    dirs = get_dirs()
    total_shots = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    # 尝试加载ChatTTS
    chat = None
    try:
        import ChatTTS
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        log("ChatTTS ✓")
    except Exception as e:
        log(f"ChatTTS: {e}")

    edge_available = False
    try:
        import edge_tts
        edge_available = True
        log("edge-tts 备用 ✓")
    except:
        pass

    EDGE_VOICES = {
        "xiaoming": "zh-CN-YunxiNeural",
        "xiaoli": "zh-CN-XiaoxiaoNeural",
        "boss_wang": "zh-CN-YunjianNeural",
        "narrator": "zh-CN-YunxiNeural"
    }

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            shot_id = shot["shot_id"]
            output_path = f"{dirs['audio']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{shot_id}.wav"

            if os.path.exists(output_path):
                continue

            char = shot.get("character", "narrator")
            dialogue = shot.get("dialogue", "")
            narration = shot.get("narration", "")
            emotion = shot.get("emotion", "calm")
            duration = shot.get("duration_seconds", 3)
            text = dialogue or narration

            if not text:
                _save_silence(output_path, float(duration))
                continue

            # 情绪调速
            voice = VOICE_PARAMS.get(char, VOICE_PARAMS["narrator"]).copy()
            voice["speed"] = voice.get("speed", 1.0) * EMOTION_SPEED.get(emotion, 1.0)

            success = False

            # ChatTTS
            if chat is not None:
                try:
                    wavs = chat.infer([text])
                    if wavs and len(wavs) > 0:
                        import torchaudio
                        audio = wavs[0]
                        if isinstance(audio, torch.Tensor):
                            torchaudio.save(output_path, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                        success = True
                except:
                    pass

            # edge-tts
            if not success and edge_available:
                try:
                    import asyncio
                    voice_name = EDGE_VOICES.get(char, "zh-CN-YunxiNeural")

                    async def _tts():
                        comm = edge_tts.Communicate(text, voice_name)
                        await comm.save(output_path)

                    asyncio.run(_tts())
                    success = True
                except:
                    pass

            if not success:
                _save_silence(output_path, float(duration))

            log(f"  [{count}/{total_shots}] {shot_id} ({char}) {'✓' if success else '静音'}")

    log("配音生成完成")


def _save_silence(output_path, duration):
    """保存静音WAV"""
    try:
        import wave, struct
        sr = AUDIO_SAMPLE_RATE
        n = int(sr * duration)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'w') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            for _ in range(n):
                w.writeframes(struct.pack('<h', 0))
    except:
        run_cmd(
            f'ffmpeg -y -f lavfi -i "anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono" '
            f'-t {duration} -acodec pcm_s16le "{output_path}" 2>/dev/null'
        )


# ============================================================
# Step 6: 剪辑合成 (FFmpeg)
# ============================================================

def step6_compose(storyboard):
    """合成最终视频"""
    log("=" * 50)
    log("Step 6: 剪辑合成 (FFmpeg)")
    log("=" * 50)

    dirs = get_dirs()

    # 1. SRT字幕
    srt_path = f"{dirs['final']}/ep{EPISODE_NUM:02d}.srt"
    total_dur = _make_srt(storyboard, srt_path)

    # 2. 收集视频
    video_list = []
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            vp = f"{dirs['videos']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{sid}.mp4"
            gp = vp.replace(".mp4", ".gif")
            if os.path.exists(vp):
                video_list.append(vp)
            elif os.path.exists(gp):
                run_cmd(f'ffmpeg -y -i "{gp}" -c:v libx264 -pix_fmt yuv420p '
                        f'-movflags +faststart "{vp}" 2>/dev/null')
                if os.path.exists(vp):
                    video_list.append(vp)

    # 3. 收集音频
    audio_list = []
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            ap = f"{dirs['audio']}/ep{EPISODE_NUM:02d}_{scene['scene_id']}_{sid}.wav"
            if os.path.exists(ap):
                audio_list.append(ap)

    log(f"视频: {len(video_list)} | 音频: {len(audio_list)}")
    if not video_list:
        log("[ERROR] 没有视频片段")
        return

    # 4. 拼接
    concat_v = f"{dirs['final']}/_video.mp4"
    concat_a = f"{dirs['final']}/_audio.wav"
    _concat_files(video_list, concat_v, "video")
    if audio_list:
        _concat_files(audio_list, concat_a, "audio")
    else:
        _save_silence(concat_a, total_dur)

    # 5. 最终合成
    final = f"{dirs['final']}/episode_{EPISODE_NUM:02d}_final.mp4"
    cmd = (
        f'ffmpeg -y -i "{concat_v}" -i "{concat_a}" '
        f'-vf "subtitles=\'{srt_path}\':force_style=\'FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2\'" '
        f'-c:v libx264 -crf 20 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -ar 44100 -ac 2 '
        f'-shortest -movflags +faststart "{final}"'
    )
    run_cmd(cmd, timeout=300)

    if os.path.exists(final):
        size = os.path.getsize(final) / 1e6
        log(f"最终视频: {final} ({size:.1f} MB)")
    else:
        # fallback 无字幕
        cmd2 = (
            f'ffmpeg -y -i "{concat_v}" -i "{concat_a}" '
            f'-c:v libx264 -crf 20 -pix_fmt yuv420p '
            f'-c:a aac -b:a 128k -ar 44100 -ac 2 '
            f'-shortest -movflags +faststart "{final}"'
        )
        run_cmd(cmd2, timeout=300)
        if os.path.exists(final):
            size = os.path.getsize(final) / 1e6
            log(f"最终视频(无硬字幕): {final} ({size:.1f} MB)")

    # 清理
    for f in [concat_v, concat_a]:
        if os.path.exists(f):
            os.remove(f)

    log("剪辑合成完成")


def _make_srt(storyboard, srt_path):
    """生成SRT"""
    lines = []
    idx = 1
    t = 0.0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            dur = shot.get("duration_seconds", 3)
            text = shot.get("subtitle") or shot.get("dialogue") or ""
            if text:
                lines.extend([str(idx), f"{seconds_to_srt_time(t)} --> {seconds_to_srt_time(t+dur)}", text, ""])
                idx += 1
            t += dur
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"SRT: {idx-1}条 | {t:.1f}s")
    return t


def _concat_files(file_list, output, media_type):
    """拼接媒体文件"""
    list_file = output + ".list"
    with open(list_file, "w") as f:
        for p in file_list:
            f.write(f"file '{p}'\n")
    if media_type == "video":
        cmd = (f'ffmpeg -y -f concat -safe 0 -i "{list_file}" '
               f'-c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "{output}"')
    else:
        cmd = (f'ffmpeg -y -f concat -safe 0 -i "{list_file}" '
               f'-acodec pcm_s16le -ar {AUDIO_SAMPLE_RATE} -ac 1 "{output}"')
    run_cmd(cmd, timeout=300)
    if os.path.exists(list_file):
        os.remove(list_file)


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

    # 依赖检查
    try:
        import google.generativeai
    except ImportError:
        log("安装依赖...")
        run_cmd("pip install -q google-generativeai diffusers transformers accelerate safetensors soundfile torchaudio moviepy edge-tts")
        try:
            run_cmd("pip install -q xformers")
        except:
            pass

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        log(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB)")

    # 运行全链路
    script = step1_generate_script()
    storyboard = step2_generate_storyboard(script)
    step3_generate_images(storyboard)

    if has_gpu:
        step4_generate_videos(storyboard)
    else:
        _placeholder_videos(storyboard)

    step5_generate_audio(storyboard)
    step6_compose(storyboard)

    elapsed = (time.time() - t0) / 60
    log(f"\n{'=' * 50}")
    log(f"全部完成! 耗时: {elapsed:.1f} 分钟")
    log(f"输出: {get_dirs()['final']}/episode_{EPISODE_NUM:02d}_final.mp4")
    log(f"{'=' * 50}")


if __name__ == "__main__":
    main()
