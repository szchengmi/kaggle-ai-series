#!/usr/bin/env python3
"""Step 3: 画面生成 - SD 1.5"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI短剧画面生成")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--output-dir", default="output/images")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))

    has_gpu = torch.cuda.is_available()
    pipe = None

    if has_gpu:
        from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
        model_path = f"{dirs['models']}/stable-diffusion-v1-5"
        if not os.path.exists(model_path) or not os.listdir(model_path):
            model_path = "runwayml/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing().enable_vae_slicing().to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        log(f"SD加载完成 | {torch.cuda.get_device_name(0)}")
    else:
        log("[WARN] 无GPU，生成占位图")

    count = 0
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = sb.get("episode", 1)
            out = f"{args.output_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.png"
            os.makedirs(args.output_dir, exist_ok=True)

            if os.path.exists(out):
                continue

            if pipe is not None:
                w = max((shot["width"] // 8) * 8, 512)
                h = max((shot["height"] // 8) * 8, 512)
                gen = None
                if shot.get("seed", -1) > 0:
                    gen = torch.Generator("cuda").manual_seed(shot["seed"])
                try:
                    r = pipe(prompt=shot["prompt"], negative_prompt=shot.get("negative_prompt", ""),
                             width=w, height=h,
                             num_inference_steps=shot.get("steps", IMAGE_STEPS),
                             guidance_scale=shot.get("guidance", IMAGE_GUIDANCE),
                             generator=gen)
                    r.images[0].save(out)
                    log(f"  [{count}/{total}] {sid} ({w}x{h}) ✓")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    _save_placeholder_image(shot, out)
                    log(f"  [{count}/{total}] {sid} OOM ✗")
            else:
                _save_placeholder_image(shot, out)

            if count % 5 == 0 and has_gpu:
                torch.cuda.empty_cache()

    log("画面生成完成")


if __name__ == "__main__":
    main()
