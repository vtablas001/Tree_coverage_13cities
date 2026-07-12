from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import ee


ROOT = Path(__file__).resolve().parents[1]
CAPITALS_CSV = ROOT / "configs" / "capitales.csv"
OUT = ROOT / "outputs_analysis" / "capitales_trienios_2012"
EE_PROJECT = "plasma-kit-334617"
AREA_KM2 = 1000.0
CLOUD = 55.0
SCALE = 30
FIELD_KM2 = 105 * 68 / 1_000_000
PERIODS = ((2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023), (2024, 2026))
SELECTED_CAPITALS = (
    "bolivia_la_paz",
    "colombia_bogota",
    "guatemala_ciudad_de_guatemala",
    "honduras_tegucigalpa",
    "paraguay_asuncion",
)


@dataclass(frozen=True)
class Capital:
    slug: str
    pais: str
    capital: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Sensor:
    collection: str
    first: int
    last: int
    bands: tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class Scenario:
    name: str
    ndvi: float
    ndbi: float
    mndwi: float
    nbr: float
    swir1: float


SENSORS = (
    Sensor("LANDSAT/LT05/C02/T1_L2", 1984, 2012, ("SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7")),
    Sensor("LANDSAT/LE07/C02/T1_L2", 1999, 2026, ("SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7")),
    Sensor("LANDSAT/LC08/C02/T1_L2", 2013, 2026, ("SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7")),
    Sensor("LANDSAT/LC09/C02/T1_L2", 2021, 2026, ("SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7")),
)

STRICT = Scenario("estricto", 0.65, -0.10, 0.00, 0.40, 0.20)


def capitals() -> list[Capital]:
    with CAPITALS_CSV.open("r", encoding="utf-8", newline="") as fh:
        return [Capital(r["slug"], r["pais"], r["capital"], float(r["latitud"]), float(r["longitud"])) for r in csv.DictReader(fh)]


def init_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def aoi(c: Capital, area_km2: float) -> ee.Geometry:
    side = math.sqrt(area_km2) / 2
    dlat = side / 110.574
    dlon = side / (111.320 * math.cos(math.radians(c.lat)))
    return ee.Geometry.Rectangle([c.lon - dlon, c.lat - dlat, c.lon + dlon, c.lat + dlat], proj=None, geodesic=False)


def scale_mask(img: ee.Image, bands: tuple[str, str, str, str, str, str]) -> ee.Image:
    qa = img.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    refl = img.select(list(bands)).multiply(0.0000275).add(-0.2)
    return refl.clamp(0, 1).rename(["red", "green", "blue", "nir", "swir1", "swir2"]).updateMask(clear)


def year_collection(year: int, geom: ee.Geometry, cloud: float) -> ee.ImageCollection | None:
    merged = None
    for sensor in SENSORS:
        if sensor.first <= year <= sensor.last:
            col = (
                ee.ImageCollection(sensor.collection)
                .filterBounds(geom)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filter(ee.Filter.lt("CLOUD_COVER", cloud))
                .map(lambda img, bands=sensor.bands: scale_mask(img, bands))
            )
            merged = col if merged is None else merged.merge(col)
    return merged


def period_image(start: int, end: int, geom: ee.Geometry, cloud: float) -> tuple[ee.Image, int]:
    imgs = []
    scenes = 0
    for year in range(start, end + 1):
        col = year_collection(year, geom, cloud)
        if col is None:
            continue
        count = int(col.size().getInfo())
        scenes += count
        if count:
            imgs.append(col.median().set("year", year))
    if not imgs:
        raise RuntimeError(f"Sin escenas para {start}-{end}")
    return ee.ImageCollection.fromImages(imgs).median().clip(geom).toFloat(), scenes


def forest(img: ee.Image, s: Scenario) -> ee.Image:
    ndvi = img.normalizedDifference(["nir", "red"])
    ndbi = img.normalizedDifference(["swir1", "nir"])
    mndwi = img.normalizedDifference(["green", "swir1"])
    nbr = img.normalizedDifference(["nir", "swir2"])
    return ndvi.gte(s.ndvi).And(ndbi.lte(s.ndbi)).And(mndwi.lte(s.mndwi)).And(nbr.gte(s.nbr)).And(img.select("swir1").lte(s.swir1))


def area(mask: ee.Image, geom: ee.Geometry) -> float:
    stats = ee.Image.pixelArea().updateMask(mask).rename("area").reduceRegion(
        ee.Reducer.sum(), geom, SCALE, maxPixels=1e10, tileScale=4
    ).getInfo()
    return float(stats.get("area") or 0) / 1_000_000


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(c: Capital, area_km2: float, cloud: float) -> list[dict[str, object]]:
    print(f"\n== {c.pais} | {c.capital} ==")
    geom = aoi(c, area_km2)
    rows = []
    for start, end in PERIODS:
        label = f"{start}-{end}"
        img, scenes = period_image(start, end, geom, cloud)
        mask = forest(img, STRICT)
        area_km2_value = area(mask, geom)
        rows.append({
            "pais": c.pais,
            "capital": c.capital,
            "slug": c.slug,
            "epoca": label,
            "anio_inicio": start,
            "anio_fin": end,
            "escenario": STRICT.name,
            "escenas_landsat": scenes,
            "area_bosque_probable_km2": round(area_km2_value, 4),
            "area_bosque_probable_ha": round(area_km2_value * 100, 2),
            "area_bosque_probable_canchas": round(area_km2_value / FIELD_KM2, 0),
        })
        print(f"{label}: {area_km2_value:.2f} km2")
    first = rows[0]
    last = rows[-1]
    saldo = float(last["area_bosque_probable_km2"]) - float(first["area_bosque_probable_km2"])
    rows.append({
        "pais": c.pais,
        "capital": c.capital,
        "slug": c.slug,
        "epoca": "saldo_2012_2014_vs_2024_2026",
        "anio_inicio": "",
        "anio_fin": "",
        "escenario": STRICT.name,
        "escenas_landsat": "",
        "area_bosque_probable_km2": round(saldo, 4),
        "area_bosque_probable_ha": round(saldo * 100, 2),
        "area_bosque_probable_canchas": round(saldo / FIELD_KM2, 0),
    })
    return rows


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula trienios 2012-2026 de bosque probable en capitales seleccionadas.")
    parser.add_argument("--countries", nargs="*", default=list(SELECTED_CAPITALS))
    parser.add_argument("--area-km2", type=float, default=AREA_KM2)
    parser.add_argument("--cloud-cover", type=float, default=CLOUD)
    parser.add_argument("--ee-project", default=os.getenv("EE_PROJECT", EE_PROJECT))
    return parser.parse_args()


def main() -> None:
    parsed = args()
    init_ee(parsed.ee_project)
    all_capitals = capitals()
    wanted = set(parsed.countries)
    selected = [c for c in all_capitals if c.slug in wanted]
    missing = wanted - {c.slug for c in selected}
    if missing:
        raise ValueError(f"Capitales no encontradas: {', '.join(sorted(missing))}")
    rows = []
    for cap in selected:
        rows.extend(analyze(cap, parsed.area_km2, parsed.cloud_cover))
    output_path = OUT / "capitales_seleccionadas_trienios_2012_2026_1000km2.csv"
    write_csv(output_path, rows)
    print(f"\nCSV -> {output_path}")


if __name__ == "__main__":
    main()
