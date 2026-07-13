from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
GIFS_DIR = ROOT / "gifs"
COUNTRIES = (
    ("bolivia", "bolivia_la_paz"),
    ("colombia", "colombia_bogota"),
    ("guatemala", "guatemala_ciudad_de_guatemala"),
    ("honduras", "honduras_tegucigalpa"),
    ("paraguay", "paraguay_asuncion"),
)
REFERENCE_FRAME = 0
CURRENT_FRAME = 4
DEFAULT_OUTPUT = GIFS_DIR / "vegetacion_10_mapas_2012_2026_reduccion_rojo.gif"


def source_paths() -> list[Path]:
    top = [
        GIFS_DIR / "frentes" / f"{country}_vegetacion_timelapse_trienios_2012.gif"
        for country, _ in COUNTRIES
    ]
    bottom = [
        GIFS_DIR / "capitales" / f"{capital}_vegetacion_densa_trienios_2012.gif"
        for _, capital in COUNTRIES
    ]
    return top + bottom


def is_green(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    return green >= 80 and green > red * 1.18 and green > blue * 1.08


def mark_loss(reference: Image.Image, current: Image.Image) -> Image.Image:
    reference_rgb = reference.convert("RGB")
    current_rgb = current.convert("RGB")
    result = current_rgb.copy()

    reference_pixels = reference_rgb.load()
    current_pixels = current_rgb.load()
    result_pixels = result.load()
    width, height = result.size

    for y in range(height):
        for x in range(width):
            was_green = is_green(reference_pixels[x, y])
            is_still_green = is_green(current_pixels[x, y])
            if was_green and not is_still_green:
                result_pixels[x, y] = (220, 38, 38)

    return result


def tile_frame(source: Image.Image, frame_index: int, tile_size: int) -> Image.Image:
    source.seek(frame_index)
    return ImageOps.fit(
        source.convert("RGB"),
        (tile_size, tile_size),
        method=Image.Resampling.LANCZOS,
    )


def generate(output_path: Path, tile_size: int) -> None:
    paths = source_paths()
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan GIF fuente: {', '.join(str(path) for path in missing)}")

    sources = [Image.open(path) for path in paths]
    try:
        frame_counts = {source.n_frames for source in sources}
        if len(frame_counts) != 1:
            raise ValueError(f"Los GIF no tienen igual numero de cuadros: {sorted(frame_counts)}")

        frame_count = frame_counts.pop()
        if CURRENT_FRAME >= frame_count:
            raise ValueError(f"Los GIF fuente tienen {frame_count} cuadros; se necesita el indice {CURRENT_FRAME}.")

        reference_canvas = Image.new("RGB", (tile_size * 5, tile_size * 2), (0, 0, 0))
        current_canvas = Image.new("RGB", (tile_size * 5, tile_size * 2), (0, 0, 0))

        for source_index, source in enumerate(sources):
            reference_tile = tile_frame(source, REFERENCE_FRAME, tile_size)
            current_tile = tile_frame(source, CURRENT_FRAME, tile_size)
            current_with_loss = mark_loss(reference_tile, current_tile)

            row, column = divmod(source_index, 5)
            position = (column * tile_size, row * tile_size)
            reference_canvas.paste(reference_tile, position)
            current_canvas.paste(current_with_loss, position)

        frames = [
            reference_canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=192),
            current_canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=192),
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=(1800, 3200),
            loop=0,
            optimize=True,
            disposal=2,
        )
    finally:
        for source in sources:
            source.close()

    print(f"GIF con reduccion en rojo -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marca en rojo los pixeles que dejan de ser vegetacion verde.")
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.output, args.tile_size)
