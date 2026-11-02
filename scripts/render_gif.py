"""Render assets/demo.gif: the LEARNED policy (not the expert) on unseen layouts, back to back.

    python scripts/render_gif.py --ckpt checkpoints/act_dart_200_s0.pt --episodes 5
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copycat.env import CONTROL_HZ, MAX_STEPS, PickPlaceEnv  # noqa: E402
from copycat.evaluate import EVAL_SEED0  # noqa: E402
from copycat.policies import ChunkExecutor  # noqa: E402
from copycat.train import load_policy  # noqa: E402

KEY_COLORS = [(255, 158, 26), (38, 230, 242), (255, 89, 166), (158, 92, 255), (235, 230, 255),
              (120, 255, 150), (255, 110, 110), (180, 150, 255)]
NAMES = {"bc": "BC-MLP", "chunk_mlp": "Chunk-MLP", "act": "ACT-lite"}


def font(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_gif(frames, path, fps):
    """One shared palette and no dithering: with a static camera most pixels stay identical
    between frames, so the GIF encoder only stores the parts that move."""
    picked = frames[:: max(1, len(frames) // 12)]
    # The scene is mostly dark purple; without help median-cut spends the whole palette there and
    # the cube comes out brown. A swatch of the colors that matter reserves entries for them.
    key = [np.array(c) * k for c in KEY_COLORS for k in (0.35, 0.55, 0.75, 0.9, 1.0)]
    rows = len(picked) * frames[0].shape[0] // 4
    swatch = np.repeat(np.repeat(np.clip(key, 0, 255).astype(np.uint8)[None], rows, 0),
                       frames[0].shape[1] // len(key) + 1, 1)[:, : frames[0].shape[1]]
    sample = Image.fromarray(np.concatenate(picked + [swatch], axis=0))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0,
                 optimize=False, disposal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "act_dart_200_s0.pt"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0, help="first layout seed (unseen in training)")
    ap.add_argument("--fps", type=int, default=30, help="playback fps; control runs at 20 Hz")
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo.gif"))
    args = ap.parse_args()

    policy = load_policy(args.ckpt)
    name = NAMES[policy.kind]
    W, H = 560, 360
    env = PickPlaceEnv()
    view = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 0.13]
    cam.distance, cam.azimuth, cam.elevation = 1.18, 135, -28
    big, small = font(16), font(13)
    frames, wins = [], 0

    def frame(ep, t, status, color):
        view.update_scene(env.data, cam)
        img = Image.fromarray(view.render())
        d = ImageDraw.Draw(img)
        d.text((12, 10), f"{name} policy, not the expert", fill=(235, 230, 255), font=big)
        d.text((12, 32), f"unseen layout {ep + 1}/{args.episodes}   t={t / CONTROL_HZ:3.1f}s   "
                         f"x{args.fps / CONTROL_HZ:.1f}", fill=(180, 150, 255), font=small)
        if status:
            d.text((12, H - 30), status, fill=color, font=big)
        frames.append(np.asarray(img))

    try:
        for ep in range(args.episodes):
            obs = env.reset(args.seed0 + ep)
            executor = ChunkExecutor(policy)
            frame(ep, 0, "", None)
            ok = False
            for t in range(MAX_STEPS):
                obs, ok = env.step(executor(obs[None])[0])
                frame(ep, t + 1, "", None)
                if ok:
                    break
            wins += ok
            status, color = ("PLACED", (120, 255, 150)) if ok else ("FAILED", (255, 110, 110))
            for _ in range(round(0.8 * args.fps)):  # hold the result for a moment
                frame(ep, t + 1, status, color)
    finally:
        view.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, args.out, args.fps)
    print(f"{wins}/{args.episodes} placed; wrote {args.out} "
          f"({len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
