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
FRAME_INDICES = (0, 4)
PERIOD_LABELS = ("2012-2014", "2024-2026")
DEFAULT_OUTPUT = GIFS_DIR / "vegetacion_10_mapas_2012_2026.gif"


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
        if max(FRAME_INDICES) >= frame_count:
            raise ValueError(f"Los GIF fuente tienen {frame_count} cuadros; se necesitan indices {FRAME_INDICES}.")

        frames = []
        for frame_index in FRAME_INDICES:
            canvas = Image.new("RGB", (tile_size * 5, tile_size * 2), (0, 0, 0))
            for source_index, source in enumerate(sources):
                source.seek(frame_index)
                tile = ImageOps.fit(
                    source.convert("RGB"),
                    (tile_size, tile_size),
                    method=Image.Resampling.LANCZOS,
                )
                row, column = divmod(source_index, 5)
                canvas.paste(tile, (column * tile_size, row * tile_size))

            frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=(1800, 2800),
            loop=0,
            optimize=True,
            disposal=2,
        )
    finally:
        for source in sources:
            source.close()

    period_list = " / ".join(PERIOD_LABELS)
    print(f"GIF de dos periodos ({period_list}) -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina los diez mapas en un GIF de dos periodos.")
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.output, args.tile_size)
