#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/fill_radiation_data_LECA1.py
------------------------------------
Rellena los huecos de radiación solar en el dataset de LECA1 usando datos
de estaciones personales (PWS) de Weather Underground descargados con Selenium.

El proceso es iterativo: se ordenan las estaciones del pool por correlación
estadística con la planta y se rellenan los huecos de mayor a menor calidad
de fuente, usando una caché local para evitar descargas repetidas.

Requisitos
----------
    pip install selenium astral tqdm
    Google Chrome + ChromeDriver compatibles instalados en el sistema.

Uso
---
    python utils/fill_radiation_data_LECA1.py
"""

import json
import logging
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

# Suprimir logs de TensorFlow/ABSL si están presentes en el entorno
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Driver Selenium
# ---------------------------------------------------------------------------
def _get_driver() -> Optional[webdriver.Chrome]:
    """Inicializa ChromeDriver en modo sin interfaz grafica."""
    prefs = {
        "download.default_directory": os.getcwd(),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts = Options()
    opts.add_experimental_option("prefs", prefs)
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
# Descarga de datos de Weather Underground
# ---------------------------------------------------------------------------
def download_wu_for_dates(
    station: str, dates: List[datetime], output_csv: Path
) -> Optional[Path]:
    """
    Descarga la columna de radiacion solar de una estacion PWS de Weather
    Underground para una lista de fechas y guarda el resultado en CSV.

    Parameters
    ----------
    station : str
        Identificador de la estacion PWS (p. ej. 'IALMAZ2').
    dates : list of datetime
        Dias a descargar.
    output_csv : Path
        Ruta del CSV de salida.

    Returns
    -------
    Path si la descarga fue exitosa, None en caso contrario.
    """
    driver = _get_driver()
    if not driver:
        return None

    collected = []
    for day in tqdm(dates, desc=f"Descargando {station}", unit="dia"):
        date_str = day.strftime("%Y-%m-%d")
        url = (
            f"https://www.wunderground.com/dashboard/pws/{station}/table/"
            f"{date_str}/{date_str}/daily"
        )
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            tables = driver.find_elements(By.TAG_NAME, "table")

            df_day = None
            for t in tables:
                tmp = pd.read_html(StringIO(t.get_attribute("outerHTML")))[0]
                if "Solar" in tmp.columns:
                    df_day = tmp.copy()
                    break

            if df_day is not None:
                df_day.insert(0, "Date", date_str)
                collected.append(df_day[["Date", "Time", "Solar"]])
            else:
                logging.warning(
                    "Columna 'Solar' no encontrada para %s en %s.", station, date_str
                )

        except TimeoutException:
            logging.warning("Timeout en %s para %s.", station, date_str)
        except Exception as exc:
            logging.error("Error en %s / %s: %s", station, date_str, exc)

    driver.quit()

    if not collected:
        logging.error("No se obtuvieron datos para la estacion %s.", station)
        return None

    df_all = pd.concat(collected, ignore_index=True).dropna(subset=["Solar"])
    if df_all.empty:
        logging.warning("Datos de radiacion vacios para %s.", station)
        return None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logging.info("Datos de %s guardados en: %s", station, output_csv)
    return output_csv


# ---------------------------------------------------------------------------
# Carga y limpieza de CSV de Weather Underground
# ---------------------------------------------------------------------------
def _load_wu_csv(csv_path: Path, station_name: str) -> Optional[pd.DataFrame]:
    """
    Carga un CSV de Weather Underground, limpia la columna Solar y construye
    un indice temporal.

    Returns
    -------
    DataFrame con indice datetime y columna 'rad_<station_name>', o None.
    """
    try:
        df = pd.read_csv(csv_path).dropna(subset=["Date", "Time", "Solar"])

        df["Solar"] = (
            df["Solar"].astype(str).str.replace(" w/m²", "", regex=False).str.strip()
        )
        df["Solar"] = pd.to_numeric(df["Solar"], errors="coerce")

        df["timestamp"] = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            format="%Y-%m-%d %I:%M %p",
            errors="coerce",
        )
        df = df.dropna(subset=["timestamp", "Solar"])
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

        col = f"rad_{station_name}"
        df = df.rename(columns={"Solar": col})[[col]]
        df = df[~df.index.duplicated(keep="first")].dropna().sort_index()

        return df

    except Exception as exc:
        logging.error("No se pudo procesar %s: %s", csv_path, exc)
        return None


# ---------------------------------------------------------------------------
# Ranking de estaciones por correlacion
# ---------------------------------------------------------------------------
def determinar_orden_por_correlacion(
    df_planta: pd.DataFrame,
    stations_pool: List[str],
    sample_year: int,
    correlation_folder: Path,
    min_overlap_points: int,
    tolerance: str,
) -> List[str]:
    """
    Ordena las estaciones del pool por correlacion con la planta usando datos
    de muestra (primer dia de cada mes del anyo de referencia).

    El resultado se almacena en cache para evitar recalculos en ejecuciones
    posteriores.

    Parameters
    ----------
    df_planta : DataFrame
        Dataset de la planta con columnas 'timestamp' y 'radiation'.
    stations_pool : list of str
        Estaciones PWS candidatas.
    sample_year : int
        Anyo de referencia para el calculo de correlacion.
    correlation_folder : Path
        Directorio donde se guardan los datos de muestra y el ranking.
    min_overlap_points : int
        Minimo de puntos solapados para considerar una estacion valida.
    tolerance : str
        Tolerancia temporal para el merge (p. ej. '15min').

    Returns
    -------
    Lista de identificadores de estacion ordenada por correlacion descendente.
    """
    ranking_file = correlation_folder / "correlation_ranking.json"

    if ranking_file.exists():
        logging.info("Cargando ranking de estaciones desde cache: %s", ranking_file)
        with open(ranking_file, "r") as f:
            return json.load(f)

    logging.info("Cache de ranking no encontrado. Calculando correlaciones...")
    correlation_folder.mkdir(parents=True, exist_ok=True)

    sample_days = [datetime(sample_year, month, 1) for month in range(1, 13)]
    logging.info(
        "Dias de muestra: %s", [d.strftime("%Y-%m-%d") for d in sample_days]
    )

    df_indexed = df_planta.set_index("timestamp").sort_index()
    df_merged = (
        df_indexed[df_indexed.index.year == sample_year]
        .dropna(subset=["radiation"])[["radiation"]]
        .copy()
    )

    for station in stations_pool:
        csv_path = correlation_folder / f"sample_{station}_{sample_year}.csv"
        if not csv_path.exists():
            download_wu_for_dates(station, sample_days, csv_path)

        if csv_path.exists():
            df_station = _load_wu_csv(csv_path, station)
            if df_station is not None:
                df_merged = pd.merge_asof(
                    left=df_merged,
                    right=df_station,
                    left_index=True,
                    right_index=True,
                    direction="nearest",
                    tolerance=pd.Timedelta(tolerance),
                )

    # Filtrar por umbral de calidad
    valid_cols = []
    for col in df_merged.columns:
        if not col.startswith("rad_"):
            continue
        overlap = df_merged[["radiation", col]].dropna().shape[0]
        if overlap >= min_overlap_points:
            valid_cols.append(col)
        else:
            logging.warning(
                "Estacion %s descalificada: %d puntos solapados (minimo: %d).",
                col.replace("rad_", ""), overlap, min_overlap_points,
            )

    if not valid_cols:
        logging.error("Ninguna estacion supero el umbral de calidad. Usando orden original.")
        return stations_pool

    correlations = (
        df_merged[valid_cols + ["radiation"]]
        .corr(numeric_only=True)["radiation"]
        .drop("radiation")
        .dropna()
    )

    if correlations.empty:
        logging.error("No se pudo calcular ninguna correlacion. Usando orden original.")
        return stations_pool

    ranked = correlations.sort_values(ascending=False).index.str.replace("rad_", "").tolist()
    logging.info("Ranking de correlacion: %s", ranked)

    with open(ranking_file, "w") as f:
        json.dump(ranked, f)
    logging.info("Ranking guardado en: %s", ranking_file)

    return ranked


# ---------------------------------------------------------------------------
# Calculo de horas solares
# ---------------------------------------------------------------------------
def add_sun_times(
    df: pd.DataFrame, lat: float, lon: float, tz: str
) -> pd.DataFrame:
    """
    Añade columnas 'sunrise' y 'sunset' al DataFrame calculadas con astral.
    Si ya existen y no contienen nulos, devuelve el DataFrame sin modificar.
    """
    if "sunrise" in df.columns and "sunset" in df.columns:
        if not df[["sunrise", "sunset"]].isnull().any().any():
            return df

    location = LocationInfo("PlantaSolar", "Espana", tz, lat, lon)
    unique_dates = df["timestamp"].dt.normalize().unique()
    sun_times = {
        d: sun(location.observer, date=d, tzinfo=location.timezone)
        for d in unique_dates
    }

    df = df.copy()
    df["date_only"] = df["timestamp"].dt.normalize()
    df["sunrise"] = df["date_only"].map({d: s["sunrise"] for d, s in sun_times.items()})
    df["sunset"] = df["date_only"].map({d: s["sunset"] for d, s in sun_times.items()})
    df = df.drop(columns=["date_only"])
    return df


# ---------------------------------------------------------------------------
# Deteccion de fechas con huecos
# ---------------------------------------------------------------------------
def get_missing_dates(df: pd.DataFrame, tz: str) -> List[datetime]:
    """
    Devuelve las fechas unicas con datos de radiacion faltantes.

    Un registro se considera faltante si:
    - El valor es NaN, o
    - El valor es 0 durante las horas de sol del dia (entre sunrise y sunset).

    Requiere que el DataFrame ya contenga las columnas 'sunrise' y 'sunset'.
    """
    df_tmp = df[["timestamp", "radiation", "sunrise", "sunset"]].copy()

    # Fechas con NaN
    nan_dates = df_tmp[df_tmp["radiation"].isna()]["timestamp"].dt.normalize().unique()

    # Fechas con cero durante horas de sol
    df_tmp["timestamp_aware"] = df_tmp["timestamp"].dt.tz_localize(
        tz, nonexistent="NaT", ambiguous="NaT"
    )
    daytime_zeros = df_tmp[
        (df_tmp["radiation"] == 0)
        & (df_tmp["timestamp_aware"] >= df_tmp["sunrise"])
        & (df_tmp["timestamp_aware"] <= df_tmp["sunset"])
    ]
    zero_dates = daytime_zeros["timestamp"].dt.normalize().unique()

    all_missing = pd.to_datetime(list(set(nan_dates) | set(zero_dates)))
    return sorted(all_missing)


# ---------------------------------------------------------------------------
# Relleno desde CSV
# ---------------------------------------------------------------------------
def fill_from_csv(
    df_target: pd.DataFrame, csv_path: Path, tolerance: str, tz: str
) -> pd.DataFrame:
    """
    Rellena NaN y ceros diurnos en df_target usando datos de un CSV de
    Weather Underground con tolerancia temporal.

    Parameters
    ----------
    df_target : DataFrame
        Dataset de la planta con columnas 'timestamp', 'radiation',
        'sunrise' y 'sunset'.
    csv_path : Path
        CSV de la estacion WU a usar como fuente.
    tolerance : str
        Tolerancia temporal para el merge (p. ej. '15min').
    tz : str
        Zona horaria de la planta (p. ej. 'Europe/Madrid').

    Returns
    -------
    DataFrame con los huecos rellenos.
    """
    df = df_target.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    station_name = csv_path.stem.split("_")[1]
    df_csv = _load_wu_csv(csv_path, station_name)

    if df_csv is None or df_csv.empty:
        logging.warning("No se pudo usar %s para rellenar.", csv_path.name)
        return df_target

    df_csv = df_csv.copy()
    df_csv.index = pd.to_datetime(df_csv.index, errors="coerce")
    df_csv = df_csv[~df_csv.index.isna()]
    df_csv = df_csv[~df_csv.index.duplicated(keep="first")].sort_index()

    # Validaciones previas al merge
    for name, idx in [("df_target", df.index), ("df_csv", df_csv.index)]:
        if not idx.is_monotonic_increasing:
            raise ValueError(f"El indice de {name} no esta ordenado.")
        if idx.hasnans:
            raise ValueError(f"El indice de {name} contiene NaT.")

    df_merged = pd.merge_asof(
        left=df,
        right=df_csv,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    df_merged["timestamp_aware"] = df_merged.index.to_series().dt.tz_localize(
        tz, nonexistent="NaT", ambiguous="NaT"
    )

    is_nan = df_merged["radiation"].isna()
    is_daytime_zero = (
        (df_merged["radiation"] == 0)
        & (df_merged["timestamp_aware"] >= df_merged["sunrise"])
        & (df_merged["timestamp_aware"] <= df_merged["sunset"])
    )

    col_rad = df_csv.columns[0]
    mask = (is_nan | is_daytime_zero) & df_merged[col_rad].notna()
    filled_count = int(mask.sum())

    df_merged.loc[mask, "radiation"] = df_merged.loc[mask, col_rad]
    logging.info(
        "Rellenados %d valores con datos de %s.", filled_count, csv_path.name
    )

    return df_merged.drop(columns=[col_rad, "timestamp_aware"]).reset_index()


# ---------------------------------------------------------------------------
# Informe de huecos restantes
# ---------------------------------------------------------------------------
def report_remaining(df: pd.DataFrame) -> None:
    """Registra el numero de huecos de radiacion que quedan sin rellenar."""
    n = df["radiation"].isna().sum()
    if n > 0:
        logging.warning("%d huecos de radiacion sin rellenar.", n)
        print(df[df["radiation"].isna()].head(10).to_string(index=False))
    else:
        logging.info("No quedan huecos de radiacion sin rellenar.")


# ---------------------------------------------------------------------------
# Funcion orquestadora
# ---------------------------------------------------------------------------
def Rellenar_datos_radiacion_from_wu(
    input_csv: str | Path,
    output_folder: str | Path,
    wunderground_folder: str | Path,
    correlation_folder: str | Path,
    stations_pool: List[str],
    sample_year: int,
    tolerance: str,
    min_overlap_points: int,
) -> None:
    """
    Orquesta el proceso completo de relleno de radiacion solar.

    Pasos
    -----
    1. Detecta dias con radiacion constante (sensor bloqueado) y los resetea.
    2. Ordena las estaciones del pool por correlacion con la planta.
    3. Calcula horas de amanecer/anochecer para todo el dataset.
    4. Itera sobre las estaciones descargando datos y rellenando huecos.
    5. Guarda el CSV con la serie rellena.

    Parameters
    ----------
    input_csv : str o Path
        CSV con el dataset consolidado de la planta.
    output_folder : str o Path
        Directorio donde se guarda el CSV de salida.
    wunderground_folder : str o Path
        Directorio de cache de datos de Weather Underground.
    correlation_folder : str o Path
        Directorio de cache del ranking de correlacion.
    stations_pool : list of str
        Estaciones PWS candidatas para el relleno.
    sample_year : int
        Anyo de referencia para el calculo de correlacion.
    tolerance : str
        Tolerancia temporal para el merge (p. ej. '15min').
    min_overlap_points : int
        Minimo de puntos solapados para validar una estacion.
    """
    output_folder = Path(output_folder)
    wunderground_folder = Path(wunderground_folder)
    correlation_folder = Path(correlation_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    wunderground_folder.mkdir(parents=True, exist_ok=True)
    correlation_folder.mkdir(parents=True, exist_ok=True)

    df_target = pd.read_csv(input_csv, parse_dates=["timestamp"])

    # --- Deteccion y reseteo de radiacion constante (sensor bloqueado) ------
    logging.info("Comprobando dias con radiacion constante (sensor bloqueado)...")
    daily = df_target.groupby(df_target["timestamp"].dt.date)["radiation"].agg(
        ["nunique", "mean"]
    )
    constant_days = daily[(daily["nunique"] == 1) & (daily["mean"] > 0)]

    if constant_days.empty:
        logging.info("No se detectaron dias con radiacion constante anómala.")
    else:
        logging.warning(
            "%d dias con radiacion constante detectados. Se resetean a NaN.",
            len(constant_days),
        )
        for day, stats in constant_days.iterrows():
            logging.warning("  - %s: valor constante = %.2f", day, stats["mean"])
        df_target.loc[
            df_target["timestamp"].dt.date.isin(constant_days.index), "radiation"
        ] = pd.NA

    # --- Ranking de estaciones ----------------------------------------------
    if not stations_pool:
        logging.warning("Lista de estaciones vacia. No se realizara ningun relleno.")
        ordered_stations = []
    else:
        ordered_stations = determinar_orden_por_correlacion(
            df_target, stations_pool, sample_year,
            correlation_folder, min_overlap_points, tolerance,
        )

    # --- Calculo de horas solares -------------------------------------------
    logging.info("Calculando horas de amanecer y anochecer...")
    df_target = add_sun_times(df_target, lat=41.76, lon=-2.46, tz="Europe/Madrid")

    # --- Bucle de relleno ---------------------------------------------------
    logging.info("Iniciando relleno con el orden: %s", ordered_stations)

    for station in ordered_stations:
        missing_days = get_missing_dates(df_target, tz="Europe/Madrid")
        if not missing_days:
            logging.info("No quedan huecos por rellenar.")
            break

        logging.info(
            "Huecos restantes: %d dias. Probando estacion: %s",
            len(missing_days), station,
        )

        csv_path = wunderground_folder / f"wu_{station}_ALL.csv"
        if not csv_path.exists():
            download_wu_for_dates(station, missing_days, csv_path)

        if csv_path.exists():
            df_target = fill_from_csv(df_target, csv_path, tolerance, tz="Europe/Madrid")

    # --- Guardado -----------------------------------------------------------
    df_target = df_target.drop(columns=["sunrise", "sunset"], errors="ignore")
    output_file = output_folder / "Datos_LECA1_Rellenos.csv"
    df_target.to_csv(output_file, index=False)
    logging.info("Serie rellena guardada en: %s", output_file)

    report_remaining(df_target)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

    Rellenar_datos_radiacion_from_wu(
        input_csv=DATA_DIR / "Datos_LECA1_15min.csv",
        output_folder=DATA_DIR,
        wunderground_folder=EXTERNAL_DIR / "wunderground",
        correlation_folder=EXTERNAL_DIR / "correlations",
        stations_pool=["ICUBOD2", "ILOSRB1", "IALMAZ2", "IQUINT63", "IGOLMA6", "ISORIA35"],
        sample_year=2023,
        tolerance="15min",
        min_overlap_points=48,
    )
