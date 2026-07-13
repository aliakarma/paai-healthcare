from pathlib import Path

from PIL import Image, ImageDraw


FIG = Path(__file__).resolve().parents[1] / "evaluation" / "figures"
GROUPS = ["ohiot1dm", "wesad", "ppg_dalia"]
THUMB = (520, 360)
COLS = 2


def tile_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((THUMB[0] - 20, THUMB[1] - 60))
    canvas = Image.new("RGB", THUMB, "white")
    canvas.paste(image, ((THUMB[0] - image.width) // 2, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, THUMB[1] - 28), path.name, fill=(0, 0, 0))
    return canvas


def make_sheet(group):
    files = sorted(path for path in FIG.glob(f"{group}_*.png") if "contact_sheet" not in path.name)
    rows = (len(files) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * THUMB[0], rows * THUMB[1]), "white")
    for i, path in enumerate(files):
        sheet.paste(tile_image(path), ((i % COLS) * THUMB[0], (i // COLS) * THUMB[1]))
    out = FIG / f"{group}_contact_sheet.png"
    sheet.save(out)
    return out


def main():
    for group in GROUPS:
        out = make_sheet(group)
        print(out.name)


if __name__ == "__main__":
    main()
