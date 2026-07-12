from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import joblib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rasterio.enums import Resampling
from rasterio.warp import reproject

from gee_export_grupo_a import AOIS, OUT, PERIODS, selected_periods
from rf_train_grupo_a import FEATURE_BANDS, MODELS_DIR, RESULTS_DIR, FIELD_HA, feature_indexes, pixel_size_m, read_raster, write_csv


ROOT = Path(__file__).resolve().parents[2]
GIF_DIR = OUT / "gifs"
LOG_DIR = OUT / "logs"
S3_BUCKET = "vatv-ss-timelapse"
PNG_SIZE = (1000, 1000)
COLORS = {
    "bosque_probable": (45, 106, 79),
    "no_bosque_probable": (245, 239, 230),
    "agua": (29, 126, 193),
    "sin_dato": (204, 204, 204),
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def read_aligned(path: Path, target_profile: dict, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        if (
            src.width == target_profile["width"]
            and src.height == target_profile["height"]
            and src.transform == target_profile["transform"]
            and src.crs == target_profile["crs"]
        ):
            data = src.read().astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return data
        dst = np.full((src.count, target_profile["height"], target_profile["width"]), np.nan, dtype=np.float32)
        for idx in range(1, src.count + 1):
            source = src.read(idx).astype(np.float32)
            if src.nodata is not None:
                source = np.where(source == src.nodata, np.nan, source)
            reproject(
                source=source,
                destination=dst[idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst


def pixel_area_m2(profile: dict) -> float:
    width_m, height_m = pixel_size_m(profile)
    return width_m * height_m


def save_classification(mask: np.ndarray, profile: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def classify_composite(model_bundle: dict, composite_path: Path, water_path: Path, output_path: Path) -> dict[str, object]:
    composite, profile, descriptions = read_raster(composite_path)
    features = composite[feature_indexes(descriptions)]
    water = read_aligned(water_path, profile, Resampling.nearest) if water_path.exists() else np.zeros((1, profile["height"], profile["width"]), dtype=np.float32)
    water = np.nan_to_num(water, nan=0.0)
    valid = np.all(np.isfinite(features), axis=0)
    land = valid & (water[0] < 1)

    x = np.moveaxis(features, 0, -1).reshape(-1, len(FEATURE_BANDS))
    pred = np.full(x.shape[0], 255, dtype=np.uint8)
    land_idx = np.flatnonzero(land.reshape(-1))
    pred[land_idx] = model_bundle["model"].predict(x[land_idx]).astype(np.uint8)
    mask = pred.reshape(land.shape)
    mask[(water[0] >= 1) & valid] = 2
    save_classification(mask, profile, output_path)

    px_area = pixel_area_m2(profile)
    area_bosque_km2 = float((mask == 1).sum() * px_area / 1_000_000)
    area_terrestre_km2 = float(land.sum() * px_area / 1_000_000)
    return {
        "mask": mask,
        "profile": profile,
        "area_bosque_km2": area_bosque_km2,
        "area_terrestre_km2": area_terrestre_km2,
        "pct_bosque": (area_bosque_km2 / area_terrestre_km2 * 100) if area_terrestre_km2 else 0.0,
    }


def rgb_from_composite(composite_path: Path, target_shape: tuple[int, int]) -> Image.Image:
    data, _, _ = read_raster(composite_path)
    rgb = np.moveaxis(data[:3], 0, -1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for band in range(3):
        channel = rgb[:, :, band]
        valid = np.isfinite(channel)
        if not valid.any():
            continue
        low, high = np.nanpercentile(channel[valid], [2, 98])
        if high <= low:
            high = low + 0.01
        output[:, :, band] = np.clip((channel - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(output, mode="RGB")
    if image.size != (target_shape[1], target_shape[0]):
        image = image.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
    return image


def draw_scale_bar(draw: ImageDraw.ImageDraw, x: int, y: int, km: int = 10) -> None:
    width = 190
    draw.rectangle([x, y, x + width, y + 8], fill=(20, 20, 20))
    draw.rectangle([x, y, x + width // 2, y + 8], fill=(255, 255, 255))
    draw.rectangle([x, y, x + width, y + 8], outline=(20, 20, 20), width=2)
    draw.text((x, y + 14), f"{km} km", font=font(20, True), fill=(20, 20, 20))


def render_png(mask: np.ndarray, composite_path: Path, output_path: Path, pais: str, ecorregion: str, epoca: str, area_km2: float, canchas: float) -> None:
    base = rgb_from_composite(composite_path, mask.shape)
    frame = ImageOps.fit(base, PNG_SIZE, method=Image.Resampling.BILINEAR).convert("RGBA")
    mask_view = np.array(ImageOps.fit(Image.fromarray(mask, mode="L"), PNG_SIZE, method=Image.Resampling.NEAREST))

    bosque_overlay = np.zeros((PNG_SIZE[1], PNG_SIZE[0], 4), dtype=np.uint8)
    bosque_overlay[mask_view == 1] = [45, 106, 79, 170]
    bosque_overlay[mask_view == 2] = [29, 126, 193, 145]
    bosque_overlay[mask_view == 255] = [204, 204, 204, 110]
    frame = Image.alpha_composite(frame, Image.fromarray(bosque_overlay, mode="RGBA"))

    overlay = Image.new("RGBA", PNG_SIZE, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rounded_rectangle([22, 22, 620, 118], radius=8, fill=(0, 0, 0, 150))
    draw_overlay.rounded_rectangle([680, 22, 978, 82], radius=8, fill=(0, 0, 0, 150))
    draw_overlay.rounded_rectangle([528, 860, 978, 978], radius=8, fill=(0, 0, 0, 150))
    frame = Image.alpha_composite(frame, overlay)
    draw = ImageDraw.Draw(frame)

    draw.text((44, 36), pais, font=font(34, True), fill=(255, 255, 255, 245))
    draw.text((44, 78), ecorregion, font=font(20), fill=(245, 245, 245, 235))
    draw.text((706, 36), epoca, font=font(30, True), fill=(255, 255, 255, 245))
    draw.text((552, 884), f"Bosque probable: {area_km2:.2f} km2", font=font(24, True), fill=(235, 255, 240, 245))
    draw.text((552, 922), f"Equivalente: {canchas:,.0f} canchas", font=font(23), fill=(245, 245, 245, 235))
    draw_scale_bar(draw, 44, 920)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output_path, quality=95)


def make_gif(paths: list[Path], output_path: Path) -> None:
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for path in paths]
    durations = [1200] * len(frames)
    if durations:
        durations[-1] = 2500
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)


def log_s3_error(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "s3_errors.log").open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def upload_s3(path: Path) -> None:
    try:
        import boto3

        required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Variables AWS faltantes: {', '.join(missing)}")
        boto3.client("s3").upload_file(
            str(path),
            S3_BUCKET,
            path.name,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/gif"},
        )
        print(f"S3 -> s3://{S3_BUCKET}/{path.name}")
    except Exception as exc:
        msg = f"{path}: {exc}"
        print(f"Advertencia S3: {msg}")
        log_s3_error(msg)


def model_for_bioma(bioma: str) -> dict:
    path = MODELS_DIR / f"rf_{bioma.lower()}_treecover2000.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No existe modelo para {bioma}: {path}")
    return joblib.load(path)


def infer_aoi(item, upload_gifs: bool, periods: tuple[tuple[int, int], ...], output_suffix: str) -> list[dict[str, object]]:
    print(f"\n== Inferencia {item.pais} | {item.ecorregion} ==")
    model_bundle = model_for_bioma(item.bioma)
    model_metrics = model_bundle.get("metricas", {})
    aoi_dir = OUT / item.slug
    water_path = aoi_dir / f"{item.slug}_agua_permanente.tif"
    rows = []
    frame_paths = []

    for start, end in periods:
        epoca = f"{start}-{end}"
        period_slug = f"{start}_{end}"
        composite_path = aoi_dir / f"{item.slug}_{period_slug}_composite.tif"
        classified_path = aoi_dir / f"{item.slug}_{period_slug}_bosque_probable.tif"
        png_path = aoi_dir / f"{item.slug}_{period_slug}_mapa.png"
        if not composite_path.exists():
            print(f"Advertencia: falta composite {composite_path}")
            continue

        result = classify_composite(model_bundle, composite_path, water_path, classified_path)
        area_km2 = result["area_bosque_km2"]
        area_ha = area_km2 * 100
        canchas = area_ha / FIELD_HA
        render_png(result["mask"], composite_path, png_path, item.pais, item.ecorregion, epoca, area_km2, canchas)
        frame_paths.append(png_path)

        rows.append({
            "pais": item.pais,
            "bioma": item.bioma,
            "ecorregion": item.ecorregion,
            "epoca": epoca,
            "anio_inicio": start,
            "anio_fin": end,
            "area_bosque_km2": round(area_km2, 4),
            "area_bosque_ha": round(area_ha, 2),
            "area_bosque_canchas": round(canchas, 0),
            "area_terrestre_km2": round(result["area_terrestre_km2"], 4),
            "pct_bosque": round(result["pct_bosque"], 4),
            "kappa_modelo": round(float(model_metrics.get("kappa", 0)), 4),
            "oa_modelo": round(float(model_metrics.get("overall_accuracy", 0)), 4),
            "classified_tif": str(classified_path),
            "png_mapa": str(png_path),
        })
        print(f"{item.pais} {epoca}: {area_km2:.2f} km2 de bosque probable")

    if frame_paths:
        gif_path = GIF_DIR / f"{item.slug}_bosque_timelapse{output_suffix}.gif"
        make_gif(frame_paths, gif_path)
        if upload_gifs:
            upload_s3(gif_path)
    return rows


def add_saldos(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = list(rows)
    by_pais: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_pais.setdefault(str(row["pais"]), []).append(row)
    for pais, items in by_pais.items():
        lookup = {row["epoca"]: row for row in items}
        for base, recent, label in (("1992-1996", "2022-2026", "saldo_1992_1996_vs_2022_2026"), ("2012-2016", "2022-2026", "saldo_2012_2016_vs_2022_2026")):
            if base not in lookup or recent not in lookup:
                continue
            base_area = float(lookup[base]["area_bosque_km2"])
            recent_area = float(lookup[recent]["area_bosque_km2"])
            saldo = recent_area - base_area
            output.append({
                "pais": pais,
                "bioma": lookup[recent]["bioma"],
                "ecorregion": lookup[recent]["ecorregion"],
                "epoca": label,
                "anio_inicio": "",
                "anio_fin": "",
                "area_bosque_km2": round(saldo, 4),
                "area_bosque_ha": round(saldo * 100, 2),
                "area_bosque_canchas": round((saldo * 100) / FIELD_HA, 0),
                "area_terrestre_km2": lookup[recent]["area_terrestre_km2"],
                "pct_bosque": round((saldo / base_area * 100) if base_area else 0, 4),
                "kappa_modelo": lookup[recent]["kappa_modelo"],
                "oa_modelo": lookup[recent]["oa_modelo"],
                "classified_tif": "",
                "png_mapa": "",
            })
    return output


def add_aggregates(rows: list[dict[str, object]], periods: tuple[tuple[int, int], ...]) -> list[dict[str, object]]:
    output = list(rows)
    period_rows = [row for row in rows if str(row["epoca"]) in {f"{s}-{e}" for s, e in periods}]
    by_period: dict[str, list[dict[str, object]]] = {}
    for row in period_rows:
        by_period.setdefault(str(row["epoca"]), []).append(row)
    for epoca, items in sorted(by_period.items()):
        area = sum(float(row["area_bosque_km2"]) for row in items)
        land = sum(float(row["area_terrestre_km2"]) for row in items)
        output.append({
            "pais": "AGREGADO_GRUPO_A",
            "bioma": "MIXTO",
            "ecorregion": "7 AOIs forestales Grupo A",
            "epoca": epoca,
            "anio_inicio": epoca[:4],
            "anio_fin": epoca[-4:],
            "area_bosque_km2": round(area, 4),
            "area_bosque_ha": round(area * 100, 2),
            "area_bosque_canchas": round((area * 100) / FIELD_HA, 0),
            "area_terrestre_km2": round(land, 4),
            "pct_bosque": round((area / land * 100) if land else 0, 4),
            "kappa_modelo": "",
            "oa_modelo": "",
            "classified_tif": "",
            "png_mapa": "",
        })
    return add_saldos(output)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica bosque probable Grupo A, calcula estadisticas y genera GIFs.")
    parser.add_argument("--countries", nargs="*", default=None, help="Slugs a procesar.")
    parser.add_argument("--period-mode", choices=("standard", "trienios_2012"), default="standard")
    parser.add_argument("--no-s3", action="store_true", help="No intenta subir GIFs a S3.")
    return parser.parse_args()


def main() -> None:
    parsed = args()
    wanted = set(parsed.countries or [item.slug for item in AOIS])
    selected = [item for item in AOIS if item.slug in wanted]
    missing = wanted - {item.slug for item in selected}
    if missing:
        raise ValueError(f"AOIs no encontrados: {', '.join(sorted(missing))}")
    periods = selected_periods(parsed.period_mode)
    output_suffix = "_trienios_2012" if parsed.period_mode == "trienios_2012" else ""
    rows: list[dict[str, object]] = []
    for item in selected:
        rows.extend(infer_aoi(item, upload_gifs=not parsed.no_s3, periods=periods, output_suffix=output_suffix))
    final_rows = add_aggregates(rows, periods)
    output_name = "estadisticas_grupo_a_trienios_2012.csv" if parsed.period_mode == "trienios_2012" else "estadisticas_grupo_a.csv"
    output_path = RESULTS_DIR / output_name
    write_csv(output_path, final_rows)
    print(f"\nEstadisticas -> {output_path}")


if __name__ == "__main__":
    main()
