from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import ee
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from analyze_capital_trienios_2012 import (
    AREA_KM2,
    CAPITALS_CSV,
    CLOUD,
    EE_PROJECT,
    FIELD_KM2,
    PERIODS,
    SCALE,
    SELECTED_CAPITALS,
    STRICT,
    Capital,
    aoi,
    capitals,
    forest,
    init_ee,
    period_image,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_analysis" / "capitales_trienios_2012"
PNG_SIZE = (1000, 1000)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def download_png(image: ee.Image, geom: ee.Geometry, output_path: Path, dimensions: int = 1200) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = image.getThumbURL({"region": geom, "dimensions": dimensions, "format": "png"})
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def draw_scale_bar(draw: ImageDraw.ImageDraw, x: int, y: int, km: int = 10) -> None:
    width = 190
    draw.rectangle([x, y, x + width, y + 8], fill=(20, 20, 20))
    draw.rectangle([x, y, x + width // 2, y + 8], fill=(255, 255, 255))
    draw.rectangle([x, y, x + width, y + 8], outline=(20, 20, 20), width=2)
    draw.text((x, y + 14), f"{km} km", font=font(20, True), fill=(20, 20, 20))


def render_frame(
    rgb_path: Path,
    overlay_path: Path,
    output_path: Path,
    capital: Capital,
    epoca: str,
    area_km2: float,
) -> None:
    rgb = ImageOps.fit(Image.open(rgb_path).convert("RGB"), PNG_SIZE, method=Image.Resampling.BILINEAR).convert("RGBA")
    overlay = ImageOps.fit(Image.open(overlay_path).convert("RGBA"), PNG_SIZE, method=Image.Resampling.NEAREST)
    frame = Image.alpha_composite(rgb, overlay)

    box = Image.new("RGBA", PNG_SIZE, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(box)
    box_draw.rounded_rectangle([22, 22, 660, 118], radius=8, fill=(0, 0, 0, 150))
    box_draw.rounded_rectangle([700, 22, 978, 82], radius=8, fill=(0, 0, 0, 150))
    box_draw.rounded_rectangle([500, 860, 978, 978], radius=8, fill=(0, 0, 0, 150))
    frame = Image.alpha_composite(frame, box)

    draw = ImageDraw.Draw(frame)
    draw.text((44, 36), f"{capital.capital}, {capital.pais}", font=font(32, True), fill=(255, 255, 255, 245))
    draw.text((44, 78), "Capital | AOI 1000 km2", font=font(20), fill=(245, 245, 245, 235))
    draw.text((724, 36), epoca, font=font(30, True), fill=(255, 255, 255, 245))
    draw.text((526, 884), f"Vegetacion densa probable: {area_km2:.2f} km2", font=font(23, True), fill=(235, 255, 240, 245))
    draw.text((526, 922), f"Equivalente: {area_km2 / FIELD_KM2:,.0f} canchas", font=font(23), fill=(245, 245, 245, 235))
    draw_scale_bar(draw, 44, 920)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output_path, quality=95)


def make_gif(frame_paths: list[Path], output_path: Path) -> None:
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for path in frame_paths]
    durations = [1200] * len(frames)
    if durations:
        durations[-1] = 2500
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)


def area(mask: ee.Image, geom: ee.Geometry) -> float:
    stats = ee.Image.pixelArea().updateMask(mask).rename("area").reduceRegion(
        ee.Reducer.sum(), geom, SCALE, maxPixels=1e10, tileScale=4
    ).getInfo()
    return float(stats.get("area") or 0) / 1_000_000


def process_capital(capital: Capital, area_km2: float, cloud: float, overwrite: bool) -> None:
    geom = aoi(capital, area_km2)
    out_dir = OUT / capital.slug
    frames_dir = out_dir / "frames"
    gif_path = OUT / "gifs" / f"{capital.slug}_vegetacion_densa_trienios_2012.gif"
    frame_paths: list[Path] = []

    print(f"\n== {capital.pais} | {capital.capital} ==")
    for start, end in PERIODS:
        epoca = f"{start}-{end}"
        period_slug = f"{start}_{end}"
        rgb_path = frames_dir / f"{capital.slug}_{period_slug}_rgb.png"
        overlay_path = frames_dir / f"{capital.slug}_{period_slug}_overlay.png"
        frame_path = frames_dir / f"{capital.slug}_{period_slug}_frame.png"

        if overwrite:
            for path in (rgb_path, overlay_path, frame_path):
                if path.exists():
                    path.unlink()

        img, scenes = period_image(start, end, geom, cloud)
        mask = forest(img, STRICT)
        area_km2_value = area(mask, geom)
        rgb = img.visualize(bands=["red", "green", "blue"], min=0.03, max=0.38, gamma=1.15)
        overlay = mask.selfMask().visualize(palette=["2d6a4f"], opacity=0.55)
        download_png(rgb, geom, rgb_path)
        download_png(overlay, geom, overlay_path)
        render_frame(rgb_path, overlay_path, frame_path, capital, epoca, area_km2_value)
        frame_paths.append(frame_path)
        print(f"{epoca}: {area_km2_value:.2f} km2 | {scenes} escenas")

    make_gif(frame_paths, gif_path)
    print(f"GIF -> {gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera GIFs trienales de vegetacion densa probable para capitales seleccionadas.")
    parser.add_argument("--countries", nargs="*", default=list(SELECTED_CAPITALS))
    parser.add_argument("--area-km2", type=float, default=AREA_KM2)
    parser.add_argument("--cloud-cover", type=float, default=CLOUD)
    parser.add_argument("--ee-project", default=os.getenv("EE_PROJECT", EE_PROJECT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    init_ee(parsed.ee_project)
    wanted = set(parsed.countries)
    selected = [c for c in capitals() if c.slug in wanted]
    missing = wanted - {c.slug for c in selected}
    if missing:
        raise ValueError(f"Capitales no encontradas: {', '.join(sorted(missing))}")
    for cap in selected:
        process_capital(cap, parsed.area_km2, parsed.cloud_cover, parsed.overwrite)


if __name__ == "__main__":
    main()
