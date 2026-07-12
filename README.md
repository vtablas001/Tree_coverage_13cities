# Geospatial Vegetation Remote Sensing Analytics across Latin American Biomes and Cities

*Independent Applied AI and Data Science Project*

This repository presents a lightweight version of a geospatial analytics project focused on vegetation density, forest cover change, and urban/peri-urban environmental patterns across selected Latin American territories.

The project combines satellite remote sensing, geospatial data engineering, spectral vegetation indicators, and machine learning to compare vegetation dynamics across two complementary geographies:

- **Forest and biome frontiers**, where land-use change and deforestation pressures are more visible.
- **Capital city regions**, where vegetation density is shaped by urban expansion, peri-urban growth, topography, water bodies, and land management.

The goal is not to produce an official land-cover inventory. Instead, this project demonstrates how applied AI and geospatial analytics can translate complex satellite data into interpretable visual and analytical products for environmental monitoring, urban resilience, and evidence-based decision-making.

## Project Highlights

- Developed a comparative geospatial analysis of vegetation density and spatial distribution across Latin American biomes and urban areas.
- Used remotely sensed and georeferenced Landsat data to produce comparable vegetation indicators across territories and time periods.
- Explored the application of spatial analytics and machine learning to support urban resilience, environmental monitoring, and policy-oriented interpretation.
- Translated complex geospatial information into GIFs, summary tables, model metrics, and concise visual narratives for non-technical audiences.

## What Is Included

This is a **lightweight repository**. It includes final outputs and reproducible scripts, but excludes heavy raw data and local model artifacts.

```text
.
|-- data/
|   |-- capitales_seleccionadas_trienios_2012_2026_1000km2.csv
|   |-- estadisticas_grupo_a_trienios_2012.csv
|   |-- metricas_modelos_grupo_a.csv
|   |-- importancia_variables_seco.csv
|   |-- importancia_variables_humedo_norte.csv
|   `-- importancia_variables_humedo_sur.csv
|-- gifs/
|   |-- capitales/
|   `-- frentes/
|-- scripts/
|   |-- gee_export_grupo_a.py
|   |-- rf_train_grupo_a.py
|   |-- inferencia_grupo_a.py
|   |-- analyze_capital_trienios_2012.py
|   `-- generate_capital_trienio_gifs.py
`-- README.md
```

The repository does **not** include:

- Raw GeoTIFF downloads
- Google Earth Engine credentials
- Local virtual environments
- Large intermediate raster files
- Trained model binaries

## Study Design

The analysis uses Areas of Interest (AOIs) of approximately **1,000 km2**. Two types of AOIs were studied:

1. **Forest frontier / biome AOIs**
   - Selected to better observe vegetation and forest-cover change in regions where land-use pressure is more environmentally meaningful.

2. **Capital city AOIs**
   - Centered around selected Latin American capitals to evaluate urban and peri-urban dense vegetation patterns.

All locations were analyzed using comparable multi-year time windows.

## Time Periods

The final analysis uses triennial windows starting in 2012:

| Period | Label |
| --- | --- |
| 2012-2014 | Baseline period |
| 2015-2017 | Intermediate period |
| 2018-2020 | Intermediate period |
| 2021-2023 | Recent period |
| 2024-2026 | Latest period |

For the Paraguay / Chaco case, **2015-2017** is used as the preferred baseline in the comparative reading because it produced a more consistent visual and analytical reference for that AOI.

## Geographic Coverage

### Forest Fronts and Biomes

| Country | Forest front / biome | Interpretation |
| --- | --- | --- |
| Bolivia | Chiquitano dry forests | Strong vegetation/forest reduction signal |
| Colombia | Northwestern Amazon moist forests | Moderate reduction signal |
| Guatemala | Peten-Veracruz moist forests | Strong reduction signal |
| Honduras | Central American Atlantic moist forests | Strong reduction signal |
| Paraguay | Western Chaco dry forests | Strong reduction signal |

### Capital City Regions

| Country | Capital |
| --- | --- |
| Bolivia | La Paz |
| Colombia | Bogota |
| Guatemala | Guatemala City |
| Honduras | Tegucigalpa |
| Paraguay | Asuncion |

Capital AOIs should be interpreted as **local urban/peri-urban vegetation studies**, not as national deforestation estimates.

## Remote Sensing Inputs

The workflow uses Landsat imagery through Google Earth Engine and derives a set of spectral indices commonly used in vegetation, built-up area, water, and disturbance analysis.

| Indicator | What it helps measure |
| --- | --- |
| NDVI | Vegetation greenness and photosynthetic vigor |
| NDBI | Built-up surfaces, bare soil, and urban signal |
| MNDWI / NDWI | Water bodies and surface water influence |
| NBR | Vegetation disturbance, burn signal, and canopy condition |
| SWIR1 | Moisture, dry soil, and forest/non-forest separation |

These variables were used as model inputs for the Random Forest classification workflow.

## Machine Learning Workflow

The project uses Random Forest classifiers trained with spatially referenced features derived from satellite imagery.

Reference labels were derived from the **Hansen Global Forest Change** dataset, using tree-cover thresholds adapted by biome type.

Three regional Random Forest models were trained:

| Model | Regions used | Hansen tree cover threshold | Kappa | F1 score |
| --- | --- | ---: | ---: | ---: |
| Dry forest model | Bolivia + Paraguay | 40% | 0.8836 | 0.9128 |
| Northern moist forest model | Guatemala + Honduras + Nicaragua | 50% | 0.8400 | 0.9425 |
| Southern moist forest model | Colombia + Panama | 50% | 0.6396 | 0.9763 |

Validation used spatial separation by blocks to reduce over-optimistic accuracy estimates caused by spatial autocorrelation.

## Main Results

Across the five selected forest front AOIs, the analysis found an approximate net reduction of:

| Metric | Value |
| --- | ---: |
| Net reduction | 635.13 km2 |
| Equivalent area | 63,513 hectares |
| Soccer-field equivalent | 88,955 professional soccer fields |

This aggregated result is intended as a comparative analytical signal across the selected AOIs, not as an official country-level estimate.

## Complete GIF Gallery

The repository includes the full lightweight visual set used for communication and review: **5 forest/biome front GIFs** and **5 capital city GIFs**.

### Forest Fronts and Biomes

| Country | Biome / forest front | Timelapse |
| --- | --- | --- |
| Bolivia | Chiquitano dry forests | ![Bolivia forest front](gifs/frentes/bolivia_bosque_timelapse_trienios_2012.gif) |
| Colombia | Northwestern Amazon moist forests | ![Colombia forest front](gifs/frentes/colombia_bosque_timelapse_trienios_2012.gif) |
| Guatemala | Peten-Veracruz moist forests | ![Guatemala forest front](gifs/frentes/guatemala_bosque_timelapse_trienios_2012.gif) |
| Honduras | Central American Atlantic moist forests | ![Honduras forest front](gifs/frentes/honduras_bosque_timelapse_trienios_2012.gif) |
| Paraguay | Western Chaco dry forests | ![Paraguay forest front](gifs/frentes/paraguay_bosque_timelapse_trienios_2012.gif) |

### Capital Cities

| Country | Capital city | Timelapse |
| --- | --- | --- |
| Bolivia | La Paz | ![La Paz dense vegetation](gifs/capitales/bolivia_la_paz_vegetacion_densa_trienios_2012.gif) |
| Colombia | Bogota | ![Bogota dense vegetation](gifs/capitales/colombia_bogota_vegetacion_densa_trienios_2012.gif) |
| Guatemala | Guatemala City | ![Guatemala City dense vegetation](gifs/capitales/guatemala_ciudad_de_guatemala_vegetacion_densa_trienios_2012.gif) |
| Honduras | Tegucigalpa | ![Tegucigalpa dense vegetation](gifs/capitales/honduras_tegucigalpa_vegetacion_densa_trienios_2012.gif) |
| Paraguay | Asuncion | ![Asuncion dense vegetation](gifs/capitales/paraguay_asuncion_vegetacion_densa_trienios_2012.gif) |

## How to Interpret the Outputs

The GIFs are designed for fast visual comparison across time. They help identify broad spatial patterns such as:

- Persistent vegetation areas
- Areas of apparent reduction
- Areas of apparent recovery or seasonal vegetation signal
- Urban/peri-urban differences between capital regions
- Stronger forest-loss signals in frontier AOIs than in capital-centered AOIs

The CSV files provide the quantitative layer behind the visual outputs.

## Methodological Notes and Limitations

This project is a technical and analytical prototype. It should be interpreted carefully:

- Landsat spatial resolution limits the detection of small patches and narrow urban vegetation corridors.
- Cloud, shadow, water, and seasonal effects can affect spectral classification.
- Capital AOIs may show stable or increasing dense vegetation even when national forest loss is occurring elsewhere.
- Urban vegetation is not the same as intact forest.
- Hansen-derived labels are useful for training and validation, but they are not a substitute for field verification.
- Results are best understood as comparable indicators and decision-support signals, not official land-cover statistics.

## Reproducibility

The included scripts document the core workflow:

| Script | Purpose |
| --- | --- |
| `scripts/gee_export_grupo_a.py` | Exports satellite composites for the selected forest/biome AOIs |
| `scripts/rf_train_grupo_a.py` | Trains Random Forest models by regional group |
| `scripts/inferencia_grupo_a.py` | Applies trained models and computes forest-cover metrics |
| `scripts/analyze_capital_trienios_2012.py` | Computes dense vegetation metrics for selected capital AOIs |
| `scripts/generate_capital_trienio_gifs.py` | Generates GIF outputs for capital time series |

To fully reproduce the project, the user must configure Google Earth Engine access locally and regenerate the excluded raster/model artifacts.

## Acknowledgement

This project was made possible through access to Google Earth Engine under a **Google Earth Engine non-profit license** granted for this work. I am grateful for that access, which enabled local, applied research using large-scale satellite data and helped make this independent environmental analytics project feasible.

## Project Positioning

This project demonstrates an applied AI and data science workflow for geospatial environmental analytics in Latin America. It bridges technical remote sensing methods and policy-oriented communication by converting satellite imagery into interpretable metrics, visual products, and comparative territorial insights.
