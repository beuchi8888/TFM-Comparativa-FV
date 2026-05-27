#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/add_openmeteo_temperature.py
------------------------------------
Descarga datos horarios de temperatura del reanálisis ERA5 a traves de la
API de Open-Meteo y los integra con el dataset de produccion de LECA1,
interpolando a resolucion de 15 minutos.

Open-Meteo no requiere clave de acceso para el endpoint de archivo historico.

Uso
---
    python utils/add_openmeteo_temperature.py
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
# Coordenadas de la planta LECA1 (Lubia, Soria)
LATITUDE = 41.68
LONGITUDE = -2.53


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def download_openmeteo_data(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    raw_output_path: Path,
) -> pd.DataFrame | None:
    """
    Descarga temperatura horaria de Open-Meteo ERA5 para un rango de fechas.

    Parameters
    ----------
    start_date, end_date : str
        Fechas en formato 'YYYY-MM-DD'.
    latitude, longitude : float
        Coordenadas de la ubicacion.
    raw_output_path : Path
        Ruta donde se guardan los datos brutos descargados.

    Returns
    -------
    pd.DataFrame o None si la descarga falla.
    """
    api_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
    }

    logging.info("Descargando datos de Open-Meteo (%s a %s)...", start_date, end_date)

    try:
        resp = requests.get(api_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            logging.error("La respuesta de la API no contiene datos horarios.")
            return None

        df = pd.DataFrame(data["hourly"])

        # Verificacion de completitud
        expected = pd.date_range(
            start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq="h"
        )
        n_expected, n_actual = len(expected), len(df)
        if n_actual == n_expected:
            logging.info("Descarga completa: %d registros horarios.", n_actual)
        else:
            logging.warning(
                "Se esperaban %d registros, se obtuvieron %d (faltan %d).",
                n_expected, n_actual, n_expected - n_actual,
            )

        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_output_path, index=False)
        logging.info("Datos brutos guardados en: %s", raw_output_path)
        return df

    except requests.exceptions.RequestException as exc:
        logging.error("Error en la peticion a Open-Meteo: %s", exc)
    except Exception as exc:
        logging.error("Error al procesar la respuesta: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Integracion con el dataset de planta
# ---------------------------------------------------------------------------
def process_and_merge(
    df_planta_path: Path,
    df_weather: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame | None:
    """
    Interpola la temperatura horaria a 15 minutos y la une con el dataset
    de planta mediante un join por indice temporal.

    Parameters
    ----------
    df_planta_path : Path
        CSV de la planta con columna 'timestamp'.
    df_weather : pd.DataFrame
        DataFrame con columnas 'time' y 'temperature_2m' (salida de Open-Meteo).
    output_path : Path
        Ruta del CSV de salida.
    """
    if df_weather is None or df_weather.empty:
        logging.error("El DataFrame del clima esta vacio.")
        return None

    # Preparar datos meteorologicos
    df_weather = df_weather.rename(
        columns={"time": "timestamp", "temperature_2m": "T_ambiente"}
    )
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
    df_weather = df_weather.set_index("timestamp")

    # Cargar datos de planta
    try:
        df_planta = pd.read_csv(df_planta_path, parse_dates=["timestamp"])
    except FileNotFoundError:
        logging.error("Archivo de planta no encontrado: %s", df_planta_path)
        return None

    df_planta = df_planta.set_index("timestamp")

    # Interpolar temperatura a 15 minutos
    df_weather_15min = df_weather.resample("15min").interpolate(method="time")

    # Union por indice (left join: se conservan todos los registros de planta)
    df_merged = df_planta.join(df_weather_15min, how="left")
    df_merged["T_ambiente"] = df_merged["T_ambiente"].interpolate(method="time")
    df_final = df_merged.reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, float_format="%.2f")
    logging.info("Archivo guardado en: %s", output_path)

    print("\nVista previa:")
    print(df_final[["timestamp", "radiation", "T_ambiente"]].head().to_string(index=False))
    print(df_final[["timestamp", "radiation", "T_ambiente"]].tail().to_string(index=False))

    return df_final


def add_temperature_from_openmeteo(
    input_csv: Path,
    output_csv: Path,
    openmeteo_cache_csv: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Funcion orquestadora: descarga (o carga desde cache) la temperatura de
    Open-Meteo y la incorpora al dataset de produccion.

    Parameters
    ----------
    input_csv : Path
        Dataset de planta limpio.
    output_csv : Path
        Dataset de planta con temperatura incorporada.
    openmeteo_cache_csv : Path
        Ruta del CSV de datos brutos de Open-Meteo.
    start_date, end_date : str
        Rango de fechas en formato 'YYYY-MM-DD'.
    """
    # Usar cache si ya existe
    if openmeteo_cache_csv.exists():
        logging.info("Cargando cache de Open-Meteo: %s", openmeteo_cache_csv)
        df_weather = pd.read_csv(openmeteo_cache_csv)
    else:
        df_weather = download_openmeteo_data(
            start_date=start_date,
            end_date=end_date,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            raw_output_path=openmeteo_cache_csv,
        )

    if df_weather is None:
        logging.error("No se pudieron obtener datos meteorologicos.")
        return None

    return process_and_merge(
        df_planta_path=input_csv,
        df_weather=df_weather,
        output_path=output_csv,
    )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    CACHE_DIR = PROJECT_ROOT / "data" / "external" / "openmeteo"

    add_temperature_from_openmeteo(
        input_csv=DATA_DIR / "Datos_LECA1_Limpio.csv",
        output_csv=DATA_DIR / "Datos_LECA1_Con_Temperatura_OpenMeteo.csv",
        openmeteo_cache_csv=CACHE_DIR / "openmeteo_raw.csv",
        start_date="2022-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )
