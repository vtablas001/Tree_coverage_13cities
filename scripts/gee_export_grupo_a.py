from __future__ import annotations

import argparse
import io
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import ee
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "grupo_a"
EE_PROJECT = "plasma-kit-334617"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
JRC_WATER = "JRC/GSW1_4/GlobalSurfaceWater"
AREA_KM2 = 1000.0
CLOUD = 20.0
SCALE = 30
CRS = "EPSG:4326"
DOWNLOAD_TIMEOUT_SECONDS = 1800

PERIODS = (
    (1992, 1996),
    (1997, 2001),
    (2002, 2006),
    (2007, 2011),
    (2012, 2016),
    (2017, 2021),
    (2022, 2026),
)
TRIENNIAL_PERIODS_2012 = ((2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023), (2024, 2026))


@dataclass(frozen=True)
class AoiGrupoA:
    pais: str
    slug: str
    bioma: str
    ecorregion: str
    lat: float
    lon: float


@dataclass(frozen=True)
class SensorSpec:
    nombre: str
    collection: str
    bands: tuple[str, str, str, str, str, str]


AOIS = (
    AoiGrupoA("Bolivia", "bolivia", "SECO", "Vegetación seca chiquitana", -17.50, -62.30),
    AoiGrupoA("Colombia", "colombia", "HUMEDO_SUR", "Vegetación húmeda amazónica del noroeste", 1.80, -74.50),
    AoiGrupoA("Guatemala", "guatemala", "HUMEDO_NORTE", "Vegetación húmeda Petén-Veracruz", 16.80, -89.60),
    AoiGrupoA("Honduras", "honduras", "HUMEDO_NORTE", "Vegetación húmeda atlántica centroamericana", 15.20, -84.80),
    AoiGrupoA("Nicaragua", "nicaragua", "HUMEDO_NORTE", "Vegetación húmeda atlántica centroamericana", 13.80, -84.20),
    AoiGrupoA("Panama", "panama", "HUMEDO_SUR", "Vegetación húmeda Chocó-Darién", 8.00, -77.60),
    AoiGrupoA("Paraguay", "paraguay", "SECO", "Vegetación seca del Chaco occidental", -22.35, -60.04),
)

SENSORS = {
    "landsat5": SensorSpec("landsat5", "LANDSAT/LT05/C02/T1_L2", ("SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7")),
    "landsat7": SensorSpec("landsat7", "LANDSAT/LE07/C02/T1_L2", ("SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7")),
    "landsat8": SensorSpec("landsat8", "LANDSAT/LC08/C02/T1_L2", ("SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7")),
    "landsat9": SensorSpec("landsat9", "LANDSAT/LC09/C02/T1_L2", ("SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7")),
}

# Coeficientes lineales usados para aproximar Landsat 5/7 a respuesta Landsat 8.
ROY_SLOPES = [0.9372, 0.9317, 0.8850, 0.8339, 0.8639, 0.9165]
ROY_INTERCEPTS = [0.0123, 0.0123, 0.0183, 0.0448, 0.0306, 0.0116]


def init_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def select_aois(slugs: list[str] | None) -> list[AoiGrupoA]:
    if not slugs:
        return list(AOIS)
    wanted = set(slugs)
    selected = [a for a in AOIS if a.slug in wanted]
    missing = wanted - {a.slug for a in selected}
    if missing:
        raise ValueError(f"AOIs no encontrados: {', '.join(sorted(missing))}")
    return selected


def selected_periods(period_mode: str) -> tuple[tuple[int, int], ...]:
    if period_mode == "trienios_2012":
        return TRIENNIAL_PERIODS_2012
    return PERIODS


def aoi(a: AoiGrupoA, area_km2: float) -> ee.Geometry:
    side = math.sqrt(area_km2) / 2
    dlat = side / 110.574
    dlon = side / (111.320 * math.cos(math.radians(a.lat)))
    return ee.Geometry.Rectangle([a.lon - dlon, a.lat - dlat, a.lon + dlon, a.lat + dlat], proj=None, geodesic=False)


def sensor_for_year(year: int) -> SensorSpec:
    if year <= 2001:
        return SENSORS["landsat5"]
    if year <= 2011:
        return SENSORS["landsat7"]
    if year <= 2020:
        return SENSORS["landsat8"]
    return SENSORS["landsat9"]


def sensors_for_year(year: int) -> tuple[SensorSpec, ...]:
    if year >= 2021:
        # Landsat 8 sigue activo; combinarlo con Landsat 9 reduce huecos por nubes en zonas humedas.
        return (SENSORS["landsat8"], SENSORS["landsat9"])
    return (sensor_for_year(year),)


def mask_and_scale(img: ee.Image, sensor: SensorSpec) -> ee.Image:
    qa = img.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    refl = img.select(list(sensor.bands)).multiply(0.0000275).add(-0.2).rename(["red", "green", "blue", "nir", "swir1", "swir2"])
    band_mask = refl.mask().reduce(ee.Reducer.min())
    refl = refl.clamp(0, 1).updateMask(clear).updateMask(band_mask)
    if sensor.nombre in {"landsat5", "landsat7"}:
        slopes = ee.Image.constant(ROY_SLOPES)
        intercepts = ee.Image.constant(ROY_INTERCEPTS)
        refl = refl.multiply(slopes).add(intercepts).clamp(0, 1).rename(["red", "green", "blue", "nir", "swir1", "swir2"])
    return refl


def yearly_collection(year: int, geom: ee.Geometry, cloud: float) -> ee.ImageCollection:
    merged = ee.ImageCollection([])
    for sensor in sensors_for_year(year):
        col = (
            ee.ImageCollection(sensor.collection)
            .filterBounds(geom)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUD_COVER", cloud))
            .map(lambda img, s=sensor: mask_and_scale(img, s))
        )
        merged = merged.merge(col)
    return merged


def period_image(start: int, end: int, geom: ee.Geometry, cloud: float) -> tuple[ee.Image, int]:
    imgs = []
    scenes = 0
    for year in range(start, end + 1):
        col = yearly_collection(year, geom, cloud)
        count = int(col.size().getInfo())
        scenes += count
        if count:
            imgs.append(col.median().set("year", year))
        else:
            print(f"Advertencia: sin escenas Landsat utiles para {year}.")
    if not imgs:
        raise RuntimeError(f"Sin escenas para {start}-{end}")
    return ee.ImageCollection.fromImages(imgs).median().clip(geom).toFloat(), scenes


def water_mask() -> ee.Image:
    return ee.Image(JRC_WATER).select("occurrence").unmask(0).gte(80).rename("agua_permanente").toByte()


def indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("ndbi")
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("mndwi")
    nbr = img.normalizedDifference(["nir", "swir2"]).rename("nbr")
    return ee.Image.cat([img, ndvi, ndbi, mndwi, nbr]).toFloat()


def composite_with_indices(start: int, end: int, geom: ee.Geometry, cloud: float) -> tuple[ee.Image, int]:
    img, scenes = period_image(start, end, geom, cloud)
    agua = water_mask()
    comp = indices(img).updateMask(agua.Not())
    return comp, scenes


def hansen_image() -> ee.Image:
    img = ee.Image(HANSEN_ASSET)
    bands = set(img.bandNames().getInfo())
    required = {"treecover2000", "lossyear"}
    if not required.issubset(bands):
        raise RuntimeError(f"El asset Hansen no contiene las bandas requeridas: {required}")
    return img.select(["treecover2000", "lossyear"]).unmask(0).rename(["treecover2000", "lossyear"]).toFloat()


def download_geotiff(img: ee.Image, geom: ee.Geometry, output_path: Path, scale: int = SCALE, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"Ya existe, saltando descarga: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    response = None
    last_error: Exception | None = None
    for requested_scale in (scale, 45, 60):
        params = {
            "region": geom,
            "scale": requested_scale,
            "crs": CRS,
            "format": "GEO_TIFF",
            "filePerBand": False,
            "maxPixels": 1e10,
        }
        try:
            url = img.getDownloadURL(params)
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            if requested_scale != scale:
                print(f"Descarga de respaldo para {output_path.name}: escala {requested_scale} m")
            break
        except Exception as exc:
            last_error = exc
            elapsed = time.monotonic() - started
            if elapsed >= DOWNLOAD_TIMEOUT_SECONDS:
                raise TimeoutError(f"Timeout de 30 minutos exportando {output_path.name}") from exc
            print(f"Reintento fallido para {output_path.name} a escala {requested_scale} m: {exc}")
    if response is None or not response.ok:
        raise RuntimeError(f"Fallo la descarga GeoTIFF para {output_path.name}: {last_error}") from last_error

    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = [name for name in zf.namelist() if name.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise RuntimeError(f"La descarga ZIP no contiene GeoTIFF para {output_path.name}")
            output_path.write_bytes(zf.read(tif_names[0]))
    else:
        output_path.write_bytes(content)
    print(f"GeoTIFF descargado: {output_path}")


def nodata_percent(img: ee.Image, geom: ee.Geometry) -> float:
    valid = img.select("red").mask().rename("valid")
    stats = valid.unmask(0).reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=1e10, tileScale=4).getInfo()
    valid_pct = float(stats.get("valid") or 0) * 100
    return 100 - valid_pct


def export_aoi(a: AoiGrupoA, area_km2: float, cloud: float, periods: tuple[tuple[int, int], ...]) -> None:
    print(f"\n== {a.pais} | {a.ecorregion} ==")
    geom = aoi(a, area_km2)
    aoi_dir = OUT / a.slug
    hansen_path = aoi_dir / f"{a.slug}_hansen_treecover2000_lossyear.tif"
    water_path = aoi_dir / f"{a.slug}_agua_permanente.tif"
    download_geotiff(hansen_image().clip(geom), geom, hansen_path)
    download_geotiff(water_mask().clip(geom), geom, water_path, overwrite=True)

    for start, end in periods:
        epoca = f"{start}_{end}"
        output_path = aoi_dir / f"{a.slug}_{epoca}_composite.tif"
        try:
            comp, scenes = composite_with_indices(start, end, geom, cloud)
            pct_nodata = nodata_percent(comp, geom)
            if pct_nodata > 30:
                print(f"Advertencia: {a.pais} {start}-{end} tiene {pct_nodata:.1f}% de pixeles sin dato.")
            print(f"{start}-{end}: {scenes} escenas Landsat")
            download_geotiff(comp, geom, output_path)
        except TimeoutError as exc:
            print(f"Timeout: {a.pais} {start}-{end}: {exc}")
        except Exception as exc:
            print(f"Advertencia: no se pudo exportar {a.pais} {start}-{end}: {exc}")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta composites Landsat e insumos Hansen/JRC para Grupo A.")
    parser.add_argument("--countries", nargs="*", default=None, help="Slugs: bolivia colombia guatemala honduras nicaragua panama paraguay.")
    parser.add_argument("--period-mode", choices=("standard", "trienios_2012"), default="standard")
    parser.add_argument("--area-km2", type=float, default=AREA_KM2)
    parser.add_argument("--cloud-cover", type=float, default=CLOUD)
    parser.add_argument("--ee-project", default=os.getenv("EE_PROJECT", EE_PROJECT))
    return parser.parse_args()


def main() -> None:
    parsed = args()
    init_ee(parsed.ee_project)
    periods = selected_periods(parsed.period_mode)
    for item in select_aois(parsed.countries):
        export_aoi(item, parsed.area_km2, parsed.cloud_cover, periods)


if __name__ == "__main__":
    main()
