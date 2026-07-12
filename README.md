# Tree Coverage - 5 Capitals and 5 Forest Fronts

Version liviana del proyecto de analisis multitemporal de cobertura vegetal con imagenes Landsat y Google Earth Engine.

El repositorio contiene GIFs y tablas de resultados para 5 capitales y 5 frentes/biomas priorizados. No incluye GeoTIFFs, modelos entrenados, credenciales ni descargas pesadas.

## Contenido

- `gifs/frentes/`: timelapses de bosque probable en 5 frentes forestales.
- `gifs/capitales/`: timelapses de vegetacion densa en 5 capitales.
- `data/`: CSVs con estadisticas, metricas de modelos Random Forest e importancia de variables.
- `scripts/`: scripts principales usados para exportar, entrenar, inferir y generar GIFs.

## Periodos

El analisis usa trienios desde 2012:

- 2012-2014
- 2015-2017
- 2018-2020
- 2021-2023
- 2024-2026

Para Paraguay / Chaco se usa 2015-2017 como periodo base principal en la lectura comparativa, por disponibilidad y consistencia visual.

## Frentes forestales incluidos

| Pais | Frente / bioma | Lectura principal |
| --- | --- | --- |
| Bolivia | Bosques secos chiquitanos | Reduccion marcada |
| Colombia | Bosques humedos amazonicos del noroeste | Reduccion moderada |
| Guatemala | Bosques humedos Peten-Veracruz | Reduccion marcada |
| Honduras | Bosques humedos atlanticos centroamericanos | Reduccion marcada |
| Paraguay | Bosques secos del Chaco occidental | Reduccion marcada |

Resultado agregado aproximado de los 5 frentes seleccionados: reduccion neta de 635.13 km2, equivalente a 63,513 ha o unas 88,955 canchas de futbol soccer profesionales.

## Capitales incluidas

| Pais | Capital |
| --- | --- |
| Bolivia | La Paz |
| Colombia | Bogota |
| Guatemala | Ciudad de Guatemala |
| Honduras | Tegucigalpa |
| Paraguay | Asuncion |

Las capitales se analizan con AOIs urbanos/periurbanos de 1000 km2. Estos resultados no deben interpretarse como deforestacion nacional; son una lectura local de vegetacion densa alrededor de cada capital.

## Modelo

Se entrenaron 3 modelos Random Forest por tipo de region:

| Modelo | Regiones usadas | Umbral Hansen treecover2000 | Kappa | F1 |
| --- | --- | ---: | ---: | ---: |
| SECO | Bolivia + Paraguay | 40% | 0.8836 | 0.9128 |
| HUMEDO_NORTE | Guatemala + Honduras + Nicaragua | 50% | 0.8400 | 0.9425 |
| HUMEDO_SUR | Colombia + Panama | 50% | 0.6396 | 0.9763 |

Variables espectrales usadas:

- NDVI: verdor y vigor de vegetacion.
- NDBI: senal de superficie construida o suelo expuesto.
- MNDWI/NDWI: agua superficial.
- NBR: vegetacion, perturbacion y quemas.
- SWIR1: humedad, suelo seco y separacion bosque/no bosque.

## Notas metodologicas

- Fuente satelital: Landsat mediante Google Earth Engine.
- Etiquetas de referencia: Hansen Global Forest Change.
- Clasificacion: Random Forest con validacion espacial por bloques.
- Area de cada AOI: 1000 km2.
- Esta version es para revision, visualizacion y trazabilidad ligera.
