#!/usr/bin/env python3
"""Step 4: 视频生成 - AnimateDiff-Lightning"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI短剧视频生成")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--images-dir", default="output/images")
    parser.add_argument("--output-dir", default="output/videos")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))
    has_gpu = torch.cuda.is_available()
    pipe = None

    if has_gpu:
        from diffusers import StableDiffusionPipeline, MotionAdapter, EulerAncestralDiscreteScheduler
        mp = f"{dirs['models']}/stable-diffusion-v1-5"
        if not os.path.exists(mp) or not os.listdir(mp):
            mp = "runwayml/stable-diffusion-v1-5"
        motp = f"{dirs['models']}/animatediff"
        if not os.path.exists(motp) or not os.listdir(motp):
            motp = "guoyww/animatediff-motion-adapter-v1-5-2"

        adapter = MotionAdapter.from_pretrained(motp, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            mp, motion_adapter=adapter, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, beta_schedule="linear", steps_offset=1
        )
        pipe.enable_attention_slicing().enable_vae_slicing().to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        log(f"AnimateDiff ✓ | {torch.cuda.get_device_name(0)}")
    else:
        log("[WARN] 无GPU，占位视频")

    motion_hints = {
        "static": "subtle motion", "pan_left": "camera panning left",
        "pan_right": "camera panning right", "dolly_in": "camera zooming in", "dolly_out": "camera zooming out"
    }

    count = 0
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = sb.get("episode", 1)
            out = f"{args.output_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            os.makedirs(args.output_dir, exist_ok=True)

            if os.path.exists(out):
                continue

            dur = shot.get("duration_seconds", 3)
            nf = min(int(dur * VIDEO_FPS), 32)
            cm = shot.get("camera_movement", "static")
            enhanced = f"{shot['prompt']}, {motion_hints.get(cm, 'subtle motion')}, smooth animation"

            if pipe is not None:
                gen = None
                if shot.get("seed", -1) > 0:
                    gen = torch.Generator("cuda").manual_seed(shot["seed"])
                try:
                    r = pipe(prompt=enhanced, negative_prompt=shot.get("negative_prompt", ""),
                             width=VIDEO_RESOLUTION, height=VIDEO_RESOLUTION,
                             num_frames=nf, num_inference_steps=15,
                             guidance_scale=7.5, generator=gen)
                    frames = r.frames[0] if isinstance(r.frames[0], list) else r.frames
                    fd = out + "_frames"
                    os.makedirs(fd, exist_ok=True)
                    for i, f in enumerate(frames):
                        f.save(f"{fd}/frame_{i:04d}.png")
                    run_cmd(f'ffmpeg -y -framerate {VIDEO_FPS} -i "{fd}/frame_%04d.png" '
                            f'-c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "{out}" 2>/dev/null')
                    shutil.rmtree(fd, ignore_errors=True)
                    log(f"  [{count}/{total}] {sid} ({nf}f) ✓")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    _save_placeholder_video(shot, out, nf)
                    log(f"  [{count}/{total}] {sid} OOM ✗")
            else:
                _save_placeholder_video(shot, out, nf)

            if count % 3 == 0 and has_gpu:
                torch.cuda.empty_cache()

    log("视频生成完成")


if __name__ == "__main__":
    main()
