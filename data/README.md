# Estructura de datos esperada

Los datos utilizados en este trabajo corresponden a instalaciones fotovoltaicas 
reales y no pueden hacerse públicos por razones de confidencialidad.

## Estructura requerida

Para ejecutar el pipeline completo, los datos deben organizarse de la siguiente manera:

```
data/
├── raw/
│   ├── LECA1/
│   │   ├── 2022/
│   │   │   ├── 01 Enero/
│   │   │   │   ├── RED_15min_20220101.csv
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── Afrisol/
│   │   └── (misma estructura por año y mes)
│   └── E03/
│       └── (misma estructura por año y mes)
└── external/
    ├── wunderground/     ← caché de Weather Underground (generado por NB00)
    ├── openmeteo/        ← caché de Open-Meteo (generado por NB00)
    └── correlations/     ← ranking de estaciones Weather Underground (generado por NB00)
```

## Formato de los CSV brutos

Cada planta debe tener un CSV diario con al menos las siguientes columnas obligatorias:


| Columna | Tipo | Descripción |
|---|---|---|
| timestamp | datetime | Marca temporal cada 15 minutos `dd/mm/yyyy HH:MM:SS`|
| radiation | float | Irradiancia solar (W/m²) |
| power | float | Potencia producida (W) |
| T_ambiente | float | Temperatura ambiente (°C) | 

**Nota** la columna de temperatura, si no existe se puede completar en el notebook 00

## Archivos generados por el pipeline

Una vez ejecutados los notebooks, se generan automáticamente:

```
data/
├── processed/            ← datasets limpios con temperatura (NB00)
├── splits/               ← particiones train/val/test (NB02)
├── models/               ← modelos serializados (NB03–NB09)
└── results/              ← métricas, tablas y figuras (NB03–NB10)
```

Ninguna de estas carpetas se incluye en el repositorio.

## Configuración de rutas

Copia `config.example.py` como `config.py` en la raíz y ajusta 
las rutas a tu entorno local.

## Modulos
 El modulo feature_horas.py extiende features.py añadiendo soporte para targets
 de múltiples horizontes temporales (Experimento 5 - Horizontes).
 Se mantiene separado para no afectar al pipeline base.
