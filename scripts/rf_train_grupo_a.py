from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

from gee_export_grupo_a import AOIS, OUT, PERIODS


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "modelos"
RESULTS_DIR = OUT / "resultados"
FEATURE_BANDS = ("ndvi", "ndbi", "mndwi", "nbr", "swir1")
TRAIN_PERIOD = (2002, 2006)
TREECOVER_THRESHOLD = 50
TREECOVER_THRESHOLDS_BY_BIOME = {
    "SECO": 40,
    "HUMEDO_SUR": 50,
    "HUMEDO_NORTE": 50,
}
BLOCK_SIZE_KM = 10.0
FIELD_HA = 0.7140
MIN_SAMPLES_PER_CLASS = 50


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_raster(path: Path) -> tuple[np.ndarray, dict, tuple[str, ...]]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        descriptions = tuple((desc or "").lower() for desc in src.descriptions)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
    return data, profile, descriptions


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

        print(f"Alineando {path.name} a la grilla Landsat...")
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


def feature_indexes(descriptions: tuple[str, ...]) -> list[int]:
    if all(name in descriptions for name in FEATURE_BANDS):
        return [descriptions.index(name) for name in FEATURE_BANDS]
    return [6, 7, 8, 9, 4]


def pixel_size_m(profile: dict) -> tuple[float, float]:
    transform = profile["transform"]
    pixel_width = abs(float(transform[0]))
    pixel_height = abs(float(transform[4]))
    crs = profile.get("crs")
    if crs is not None and getattr(crs, "is_geographic", False):
        height = int(profile["height"])
        center_lat = float(transform[5]) + float(transform[4]) * height / 2
        return pixel_width * 111_320.0 * math.cos(math.radians(center_lat)), pixel_height * 110_574.0
    return pixel_width, pixel_height


def spatial_blocks(profile: dict, aoi_slug: str) -> np.ndarray:
    width_m, height_m = pixel_size_m(profile)
    block_cols = max(1, int(round((BLOCK_SIZE_KM * 1000) / max(width_m, 1))))
    block_rows = max(1, int(round((BLOCK_SIZE_KM * 1000) / max(height_m, 1))))
    rows = np.arange(profile["height"]) // block_rows
    cols = np.arange(profile["width"]) // block_cols
    return np.array([f"{aoi_slug}_{r}_{c}" for r in rows for c in cols], dtype=object).reshape(profile["height"], profile["width"])


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "matriz_confusion": matrix.tolist(),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision_bosque_probable": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_bosque_probable": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_bosque_probable": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def sample_class_indices(y: np.ndarray, max_samples_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if idx.size < MIN_SAMPLES_PER_CLASS:
            raise RuntimeError(f"Muestra insuficiente para clase {cls}: {idx.size} pixeles.")
        if idx.size > max_samples_per_class:
            idx = rng.choice(idx, max_samples_per_class, replace=False)
        selected.append(idx)
    output = np.concatenate(selected)
    rng.shuffle(output)
    return output


def load_aoi_training(aoi_slug: str, treecover_threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, end = TRAIN_PERIOD
    aoi_dir = OUT / aoi_slug
    composite_path = aoi_dir / f"{aoi_slug}_{start}_{end}_composite.tif"
    hansen_path = aoi_dir / f"{aoi_slug}_hansen_treecover2000_lossyear.tif"
    water_path = aoi_dir / f"{aoi_slug}_agua_permanente.tif"
    if not composite_path.exists():
        raise FileNotFoundError(f"No existe composite de entrenamiento: {composite_path}")
    if not hansen_path.exists():
        raise FileNotFoundError(f"No existe Hansen recortado: {hansen_path}")

    composite, profile, descriptions = read_raster(composite_path)
    idx = feature_indexes(descriptions)
    features = composite[idx]
    hansen = read_aligned(hansen_path, profile, Resampling.bilinear)
    water = read_aligned(water_path, profile, Resampling.nearest) if water_path.exists() else np.zeros((1, profile["height"], profile["width"]), dtype=np.float32)
    water = np.nan_to_num(water, nan=0.0)

    lossyear = hansen[1]
    max_lossyear = int(np.nanmax(lossyear))
    assert max_lossyear != 0, f"Hansen lossyear maximo es 0 en {aoi_slug}; no hay perdida registrada para validar rango."
    if max_lossyear > 25:
        raise AssertionError(f"Hansen lossyear maximo inesperado en {aoi_slug}: {max_lossyear}")

    treecover = hansen[0]
    valid = np.all(np.isfinite(features), axis=0) & np.isfinite(treecover) & (water[0] < 1)
    x = np.moveaxis(features, 0, -1).reshape(-1, len(FEATURE_BANDS))
    y = (treecover.reshape(-1) >= treecover_threshold).astype(np.uint8)
    block_ids = spatial_blocks(profile, aoi_slug).reshape(-1)
    valid_flat = valid.reshape(-1)
    return x[valid_flat], y[valid_flat], block_ids[valid_flat]


def split_blocks(block_ids: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(block_ids.tolist())), dtype=object)
    rng.shuffle(unique)
    test_count = max(1, int(round(unique.size * 0.2)))
    test_blocks = set(unique[:test_count].tolist())
    test_mask = np.array([block in test_blocks for block in block_ids], dtype=bool)
    return ~test_mask, test_mask


def train_bioma(bioma: str, aoi_slugs: list[str], max_samples_per_class: int) -> dict[str, object]:
    print(f"\n== Modelo {bioma} ==")
    treecover_threshold = TREECOVER_THRESHOLDS_BY_BIOME.get(bioma, TREECOVER_THRESHOLD)
    xs, ys, blocks = [], [], []
    for slug in aoi_slugs:
        print(f"Leyendo muestras {slug}...")
        x, y, block_ids = load_aoi_training(slug, treecover_threshold)
        xs.append(x)
        ys.append(y)
        blocks.append(block_ids)
    x_all = np.vstack(xs)
    y_all = np.concatenate(ys)
    block_all = np.concatenate(blocks)
    train_mask, test_mask = split_blocks(block_all)

    train_idx_all = np.flatnonzero(train_mask)
    test_idx_all = np.flatnonzero(test_mask)
    train_idx = train_idx_all[sample_class_indices(y_all[train_idx_all], max_samples_per_class, seed=42)]
    test_idx = test_idx_all[sample_class_indices(y_all[test_idx_all], max_samples_per_class, seed=142)]

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_all[train_idx], y_all[train_idx])
    pred = model.predict(x_all[test_idx])
    stats = metricas(y_all[test_idx], pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"rf_{bioma.lower()}_treecover2000.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_bands": FEATURE_BANDS,
            "bioma": bioma,
            "treecover_umbral_pct": treecover_threshold,
            "metricas": stats,
        },
        model_path,
    )
    print(f"Modelo guardado: {model_path}")
    print(f"{bioma}: OA={stats['overall_accuracy']:.4f} | Kappa={stats['kappa']:.4f} | F1={stats['f1_bosque_probable']:.4f}")

    importance_rows = [
        {"bioma": bioma, "variable": band, "importancia": float(value)}
        for band, value in zip(FEATURE_BANDS, model.feature_importances_)
    ]
    write_csv(RESULTS_DIR / f"importancia_variables_{bioma.lower()}.csv", importance_rows)

    return {
        "bioma": bioma,
        "aois_entrenamiento": ",".join(aoi_slugs),
        "modelo_path": str(model_path),
        "treecover_umbral_pct": treecover_threshold,
        "muestras_train": int(train_idx.size),
        "muestras_test": int(test_idx.size),
        **stats,
    }


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelos Random Forest por bioma para bosque probable Grupo A.")
    parser.add_argument("--countries", nargs="*", default=None, help="Slugs a usar para entrenamiento piloto.")
    parser.add_argument("--max-samples-per-class", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    parsed = args()
    wanted = set(parsed.countries or [item.slug for item in AOIS])
    selected = [item for item in AOIS if item.slug in wanted]
    missing = wanted - {item.slug for item in selected}
    if missing:
        raise ValueError(f"AOIs no encontrados: {', '.join(sorted(missing))}")
    groups: dict[str, list[str]] = {}
    for item in selected:
        groups.setdefault(item.bioma, []).append(item.slug)
    rows = [train_bioma(bioma, slugs, parsed.max_samples_per_class) for bioma, slugs in sorted(groups.items())]
    write_csv(RESULTS_DIR / "metricas_modelos_grupo_a.csv", rows)
    print(f"\nMetricas -> {RESULTS_DIR / 'metricas_modelos_grupo_a.csv'}")


if __name__ == "__main__":
    main()
