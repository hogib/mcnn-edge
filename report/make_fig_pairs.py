"""Render report/fig/pairs.png: frame 18 and three partners for the scene-matching example."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sonar_edge.data import load_frames

OUT = Path(__file__).parent / "fig" / "pairs.png"
PANELS = [(18, "frame 18"), (187, "frame 187: linked"),
          (151, "frame 151: same scene, via chain"), (124, "frame 124: other scene, no tire")]
COLS = {"Bottle": (255, 80, 80), "Tire": (80, 200, 255), "Wall": (160, 160, 160)}

frames = {f.index: f for f in load_frames()}
try:
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()
W, H, pad = 320, 480, 6
canvas = Image.new("RGB", (W * 4 + pad * 5, H + pad * 2 + 22), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
for k, (i, label) in enumerate(PANELS):
    img = Image.fromarray(frames[i].load()).convert("RGB")
    d = ImageDraw.Draw(img)
    for b in frames[i].boxes:
        d.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=COLS.get(b.cls, (255, 255, 0)), width=2)
        d.text((b.x + 2, b.y + 2), b.cls, fill=COLS.get(b.cls, (255, 255, 0)), font=font)
    x = pad + k * (W + pad)
    canvas.paste(img, (x, pad + 22))
    dr.text((x, pad), label, fill=(0, 0, 0), font=font)
OUT.parent.mkdir(exist_ok=True)
canvas.save(OUT)
print("wrote", OUT)
