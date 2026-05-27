#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/add_wunderground_temperature.py
---------------------------------------
Descarga datos de temperatura de Weather Underground (estaciones personales PWS)
mediante Selenium y los integra con el dataset de produccion de LECA1.

Requisitos
----------
    pip install selenium
    Google Chrome + ChromeDriver compatibles instalados en el sistema.

Uso
---
    python utils/add_wunderground_temperature.py
"""

import logging
import os
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _get_driver() -> Optional[webdriver.Chrome]:
    """Inicializa ChromeDriver en modo sin interfaz grafica."""
    opts = Options()
    opts.add_experimental_option(
        "excludeSwitches", ["enable-logging", "enable-automation"]
    )
    opts.add_argument("--headless=new")
    opts.add_argument("--log-level=3")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    service = Service(log_path=os.devnull)

    try:
        return webdriver.Chrome(service=service, options=opts)
    except WebDriverException as exc:
        logging.error("No se pudo inicializar ChromeDriver: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Descarga diaria
# ---------------------------------------------------------------------------
def _download_day(
    driver: webdriver.Chrome,
    date: datetime,
    station_id: str,
    cache_dir: Path,
) -> pd.DataFrame | None:
    """
    Descarga los datos de un dia de una estacion PWS de Weather Underground.
    Si el archivo de cache existe, lo devuelve directamente.

    Parameters
    ----------
    driver : webdriver.Chrome
        Instancia de Selenium en ejecucion.
    date : datetime
        Dia a descargar.
    station_id : str
        Identificador de la estacion PWS (p. ej. 'IALMAZ2').
    cache_dir : Path
        Directorio donde se almacena el cache por estacion.
    """
    file_path = cache_dir / f"{date.strftime('%Y-%m-%d')}.csv"
    if file_path.exists():
        logging.info("Cache encontrado: %s", file_path)
        return pd.read_csv(file_path)

    url = (
        f"https://www.wunderground.com/dashboard/pws/{station_id}/table/"
        f"{date.strftime('%Y-%m-%d')}/{date.strftime('%Y-%m-%d')}/daily"
    )
    logging.info("Descargando %s desde %s", date.strftime("%Y-%m-%d"), station_id)

    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        tables = driver.find_elements(By.TAG_NAME, "table")

        df_day = None
        for table in tables:
            tmp = pd.read_html(StringIO(table.get_attribute("outerHTML")))[0]
            if "Temperature" in tmp.columns:
                df_day = tmp.copy()
                df_day.dropna(how="all", inplace=True)
                df_day.dropna(subset=["Time"], inplace=True)
                break

        if df_day is None:
            logging.warning(
                "No se encontro columna 'Temperature' para %s en %s",
                date.strftime("%Y-%m-%d"), station_id,
            )
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        df_day.to_csv(file_path, index=False)
        time.sleep(1)
        return df_day

    except TimeoutException:
        logging.warning("Timeout en %s para %s.", station_id, date.strftime("%Y-%m-%d"))
    except Exception as exc:
        logging.error("Error en %s: %s", date.strftime("%Y-%m-%d"), exc)

    return None


# ---------------------------------------------------------------------------
# Descarga del historico completo
# ---------------------------------------------------------------------------
def get_wunderground_history(
    start_date: datetime,
    end_date: datetime,
    station_ids: list[str],
    cache_base_dir: Path,
) -> pd.DataFrame:
    """
    Descarga el historico completo iterando por dias, con fallback entre
    estaciones y cache persistente.

    Parameters
    ----------
    start_date, end_date : datetime
        Rango temporal a descargar.
    station_ids : list[str]
        Estaciones PWS a intentar en orden de prioridad para cada dia.
    cache_base_dir : Path
        Directorio raiz del cache (se crea una subcarpeta por estacion).
    """
    driver = _get_driver()
    if driver is None:
        logging.error("No se pudo inicializar el driver. Abortando.")
        return pd.DataFrame()

    all_dfs = []
    current_date = start_date

    while current_date <= end_date:
        df_day = None
        for station_id in station_ids:
            cache_dir = cache_base_dir / station_id
            df_day = _download_day(driver, current_date, station_id, cache_dir)
            if df_day is not None and not df_day.empty:
                break

        if df_day is not None and not df_day.empty:
            df_day = df_day.copy()
            df_day.insert(0, "Date", current_date.strftime("%Y-%m-%d"))
            all_dfs.append(df_day)

        current_date += timedelta(days=1)

    driver.quit()

    if not all_dfs:
        logging.error("No se obtuvieron datos. Abortando.")
        return pd.DataFrame()

    df_total = pd.concat(all_dfs, ignore_index=True)
    logging.info("Total de registros descargados: %d", len(df_total))

    # Informe de dias faltantes
    expected = pd.to_datetime(
        pd.date_range(start=start_date, end=end_date, freq="D").date
    )
    downloaded = pd.to_datetime(df_total["Date"]).unique()
    missing = expected.difference(downloaded)

    if not missing.empty:
        logging.warning("Dias sin datos disponibles en Weather Underground:")
        for d in missing:
            logging.warning("  - %s", d.strftime("%Y-%m-%d"))

    return df_total


# ---------------------------------------------------------------------------
# Integracion con el dataset de planta
# ---------------------------------------------------------------------------
def process_and_merge(
    df_planta_path: Path,
    df_weather: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame | None:
    """
    Limpia los datos de Weather Underground, convierte Fahrenheit a Celsius,
    interpola a 15 minutos y une con el dataset de planta.
    """
    if df_weather.empty:
        logging.error("El DataFrame del clima esta vacio.")
        return None

    # Construir columna timestamp
    df_weather["timestamp"] = pd.to_datetime(
        df_weather["Date"] + " " + df_weather["Time"],
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )

    # Localizar columna de temperatura
    temp_col = next(
        (c for c in df_weather.columns if "Temperature" in c), None
    )
    if temp_col is None:
        logging.error("No se encontro columna 'Temperature'.")
        return None

    # Limpiar unidades y convertir a numerico
    df_weather[temp_col] = (
        df_weather[temp_col].astype(str).str.replace(r"\s.*", "", regex=True)
    )
    df_weather[temp_col] = pd.to_numeric(df_weather[temp_col], errors="coerce")
    df_weather.dropna(subset=[temp_col], inplace=True)

    # Conversion Fahrenheit -> Celsius
    df_weather["T_ambiente"] = (df_weather[temp_col] - 32) * 5.0 / 9.0

    df_weather = df_weather[["timestamp", "T_ambiente"]].copy()
    df_weather.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    df_weather = df_weather.sort_values("timestamp").set_index("timestamp")

    # Remuestrear a 15 minutos (maximo 45 min de interpolacion)
    if not df_weather.empty:
        df_weather = df_weather.resample("15min").mean()
        df_weather = df_weather.interpolate(
            method="time", limit=3, limit_direction="both"
        )

    # Cargar datos de planta
    df_planta = pd.read_csv(df_planta_path, parse_dates=["timestamp"])
    df_planta = df_planta.set_index("timestamp")

    # Union por indice
    df_merged = df_planta.join(df_weather, how="left")

    # Diagnostico de huecos antes de interpolar
    temp_col_merged = df_merged["T_ambiente"]
    daily = temp_col_merged.groupby(temp_col_merged.index.date).agg(["size", "count"])
    daily["missing"] = daily["size"] - daily["count"]
    missing_days = daily[daily["missing"] > 0]

    if not missing_days.empty:
        significant = missing_days[missing_days["missing"] * 15 / 60 > 6]
        logging.warning(
            "Huecos de temperatura detectados: %d dias con datos faltantes, "
            "%d con mas de 6 horas.",
            len(missing_days), len(significant),
        )

    # Forzar tipo numerico antes de interpolar (evita FutureWarning)
    for col in ["radiation", "T_ambiente"]:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    df_merged = df_merged.infer_objects(copy=False)
    df_merged = df_merged.interpolate(method="time")
    df_final = df_merged.reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, float_format="%.2f")
    logging.info("Archivo guardado en: %s", output_path)

    print("\nVista previa:")
    print(df_final[["timestamp", "radiation", "T_ambiente"]].head().to_string(index=False))

    return df_final


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    CACHE_DIR = PROJECT_ROOT / "data" / "external" / "wunderground"

    STATION_IDS = ["IALMAZ2", "IGOLMA6", "ISORIA35", "ICUBOD2"]
    FECHA_INICIO = datetime(2023, 1, 1)
    FECHA_FIN = datetime.now()

    df_weather_history = get_wunderground_history(
        start_date=FECHA_INICIO,
        end_date=FECHA_FIN,
        station_ids=STATION_IDS,
        cache_base_dir=CACHE_DIR,
    )

    process_and_merge(
        df_planta_path=DATA_DIR / "Datos_LECA1_Limpio.csv",
        df_weather=df_weather_history,
        output_path=DATA_DIR / "Datos_LECA1_Con_Temperatura_WU.csv",
    )
