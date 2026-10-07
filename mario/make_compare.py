"""左右对比 GIF：harness 作者的 Jev 通关录像 vs 我们本地 StartLux-Decision-0.8B 的录像。

两份 GIF 都出自未经改动的 jev-mario harness（runs/ 目录）。本脚本只做拼接，
既不碰任何一次运行，也不碰 harness。

    python make_compare.py [倍数]      # 倍数默认 2，即放大到 512x480 再拼接
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RUNS = Path(__file__).resolve().parent / "jev-mario-main" / "runs"
LEFT = RUNS / "1-1-branch-jev-20260918-170421.gif"        # 随仓库自带：1-1 通关
RIGHT = RUNS / "1-1-branch-startlux-0.8b-20261006-163620.gif"  # 本机 0.8B：走到 x=2370 停下
OUT = RUNS / "1-1-compare-jev-vs-startlux-0.8b.gif"

SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BAND = 18 * SCALE
FPS = 30  # 与 harness 写出的帧率一致：frames[::2]，1/30 秒一帧


def frames(path):
    """逐帧产出 RGB 图（GIF 是调色板模式，必须转成 RGB 才能和其他图拼接）。"""
    im = Image.open(path)
    for f in range(im.n_frames):
        im.seek(f)
        yield im.convert("RGB")


def label_bar(width, texts, font):
    """顶部标签条：左半边写左侧来源，右半边写右侧来源，中间画一条分隔线。"""
    bar = Image.new("RGB", (width, BAND), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    for text, x, color in texts:
        d.text((x + 6 * SCALE, (BAND - font.size) // 2), text, font=font, fill=color)
    d.line([(width // 2, 0), (width // 2, BAND - 1)], fill=(200, 200, 200))
    return bar


def main():
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12 * SCALE)
    left, right = frames(LEFT), frames(RIGHT)
    w, h = 256 * SCALE, 240 * SCALE
    canvas_w = w * 2

    bar = label_bar(
        canvas_w,
        [
            ("Jev（云端，通关）", 0, (20, 20, 20)),
            ("StartLux-0.8B（本机，x=2370）", w, (20, 20, 20)),
        ],
        font,
    )

    out = []
    last = None
    for i in range(max_frames := 1200):
        try:
            lf = next(left).resize((w, h), Image.NEAREST)
        except StopIteration:
            lf = None
        try:
            rf = next(right).resize((w, h), Image.NEAREST)
        except StopIteration:
            rf = None
        if lf is None and rf is None:
            break
        # 短的那一侧冻结在最后一帧，好让两边画面同时可见
        if lf is None:
            lf = last[0]
        if rf is None:
            rf = last[1]
        last = (lf, rf)

        panel = Image.new("RGB", (canvas_w, h + BAND), (255, 255, 255))
        panel.paste(bar, (0, 0))
        panel.paste(lf, (0, BAND))
        panel.paste(rf, (w, BAND))
        out.append(panel)

    out[0].save(
        OUT,
        save_all=True,
        append_images=out[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"frames={len(out)}  canvas={canvas_w}x{h + BAND}  scale={SCALE}x")
    print(f"{OUT.name}  {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
