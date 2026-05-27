#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data.py
------------
Funciones de carga, validacion y preparacion del dataset unificado de plantas
fotovoltaicas (Afrisol, E03, LECA1).

Incluye:
- Carga y validacion de CSV por planta
- Union de multiples plantas en un unico DataFrame
- Normalizacion de potencia a per-unit
- Resumen estadistico por planta
- Particion temporal y por planta para train/validacion/test
- Limpieza conservadora basica
- Comprobacion de consistencia temporal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Columnas requeridas y opcionales
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "timestamp",
    "radiation",
    "power",
    "T_ambiente",
}

OPTIONAL_COLUMNS = {
    "day_of_year",
    "hour",
    "minute",
    "month",
    "weekday",
    "is_weekend",
    "sin_day",
    "cos_day",
    "sin_hour",
    "cos_hour",
    "Mes",
}


# ---------------------------------------------------------------------------
# Estructura de planta
# ---------------------------------------------------------------------------
@dataclass
class PlantDataset:
    """Asocia un identificador de planta con la ruta de su CSV."""
    plant_id: str
    path: str | Path


# ---------------------------------------------------------------------------
# Validacion interna
# ---------------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame, plant_id: str) -> None:
    """Lanza ValueError si faltan columnas obligatorias."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{plant_id}] Faltan columnas obligatorias: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Carga de una planta
# ---------------------------------------------------------------------------
def load_single_plant_csv(
    path: str | Path,
    plant_id: str,
    drop_duplicate_timestamps: bool = True,
) -> pd.DataFrame:
    """
    Carga el CSV de una planta, valida las columnas minimas, convierte
    timestamp, elimina duplicados y ordena cronologicamente.

    Parameters
    ----------
    path : str o Path
        Ruta al CSV de la planta.
    plant_id : str
        Identificador de la planta (p. ej. 'LECA1').
    drop_duplicate_timestamps : bool
        Si True, elimina timestamps duplicados conservando el ultimo registro.

    Returns
    -------
    pd.DataFrame con columna 'id_planta' y timestamps ordenados.

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    ValueError si faltan columnas obligatorias.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, plant_id)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df["timestamp"].notna()].copy()

    if drop_duplicate_timestamps:
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        removed = before - len(df)
        if removed > 0:
            logging.warning(
                "[%s] Eliminados %d timestamps duplicados.", plant_id, removed
            )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["id_planta"] = plant_id

    first_cols = ["id_planta", "timestamp"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    logging.info(
        "[%s] Cargados %d registros. Rango: %s -> %s",
        plant_id, len(df),
        df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Carga de multiples plantas
# ---------------------------------------------------------------------------
def load_multiple_plants(datasets: Iterable[PlantDataset]) -> pd.DataFrame:
    """
    Carga y une varias plantas en un unico DataFrame ordenado.

    Parameters
    ----------
    datasets : iterable de PlantDataset
        Cada elemento asocia un plant_id con la ruta de su CSV.

    Returns
    -------
    pd.DataFrame combinado, ordenado por planta y timestamp.

    Raises
    ------
    ValueError si no se proporciona ningun dataset.
    """
    frames = []
    for ds in datasets:
        df = load_single_plant_csv(ds.path, ds.plant_id)
        frames.append(df)

    if not frames:
        raise ValueError("No se han proporcionado datasets.")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["id_planta", "timestamp"]).reset_index(drop=True)

    logging.info(
        "Dataset unificado: %d registros | %d plantas",
        len(combined), combined["id_planta"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Normalizacion de potencia
# ---------------------------------------------------------------------------
def add_nominal_power_and_target(
    df: pd.DataFrame,
    nominal_power_map: dict[str, float],
    target_col: str = "power",
    new_target_col: str = "power_pu",
) -> pd.DataFrame:
    """
    Añade la potencia nominal por planta y crea el target normalizado
    en per-unit (power_pu = power_kW / p_nominal_kW).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset unificado con columna 'id_planta'.
    nominal_power_map : dict
        Diccionario {plant_id: potencia_nominal_kW}.
    target_col : str
        Columna de potencia en W (por defecto 'power').
    new_target_col : str
        Nombre de la columna normalizada (por defecto 'power_pu').

    Returns
    -------
    pd.DataFrame con columnas 'p_nominal_kw' y new_target_col añadidas.

    Raises
    ------
    ValueError si alguna planta no tiene potencia nominal definida.
    """
    out = df.copy()
    out["p_nominal_kw"] = out["id_planta"].map(nominal_power_map)

    missing_plants = out[out["p_nominal_kw"].isna()]["id_planta"].unique()
    if len(missing_plants) > 0:
        raise ValueError(
            f"Falta potencia nominal para: {sorted(missing_plants)}"
        )

    power_kw = out[target_col] / 1000.0
    out[new_target_col] = power_kw / out["p_nominal_kw"]

    return out


# ---------------------------------------------------------------------------
# Resumen por planta
# ---------------------------------------------------------------------------
def summarize_plants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadistico rapido por planta.

    Returns
    -------
    pd.DataFrame con columnas: id_planta, n_filas, fecha_inicio, fecha_fin,
    n_power_nan, n_radiation_nan, n_temp_nan.
    """
    summary = (
        df.groupby("id_planta")
        .agg(
            n_filas=("timestamp", "size"),
            fecha_inicio=("timestamp", "min"),
            fecha_fin=("timestamp", "max"),
            n_power_nan=("power", lambda s: int(s.isna().sum())),
            n_radiation_nan=("radiation", lambda s: int(s.isna().sum())),
            n_temp_nan=("T_ambiente", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def split_train_test_by_plant(
    df: pd.DataFrame,
    train_plants: list[str],
    test_plant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en train (plantas de entrenamiento) y test (planta objetivo).

    Raises
    ------
    ValueError si alguno de los conjuntos queda vacio.
    """
    train_df = df[df["id_planta"].isin(train_plants)].copy()
    test_df = df[df["id_planta"] == test_plant].copy()

    if train_df.empty:
        raise ValueError("El conjunto de train esta vacio.")
    if test_df.empty:
        raise ValueError("El conjunto de test esta vacio.")

    return train_df, test_df


def split_train_val_test(
    df: pd.DataFrame,
    train_plants: list[str],
    test_plant: str,
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-02-29",
    test_start: str = "2024-03-01",
    test_end: str = "2024-05-30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Particion temporal y por planta en tres conjuntos disjuntos:

    - Train:       plantas de entrenamiento, hasta train_end.
    - Validacion:  plantas de entrenamiento, entre val_start y val_end.
    - Test:        planta objetivo, entre test_start y test_end.

    Parameters
    ----------
    df : pd.DataFrame
    train_plants : list of str
        Plantas usadas para entrenar y validar.
    test_plant : str
        Planta reservada para test.
    train_end, val_start, val_end, test_start, test_end : str
        Fechas de corte en formato 'YYYY-MM-DD'.

    Returns
    -------
    tuple (train_df, val_df, test_df).

    Raises
    ------
    ValueError si alguno de los tres conjuntos queda vacio.
    """
    train_df = df[
        df["id_planta"].isin(train_plants)
        & (df["timestamp"] <= train_end)
    ].copy()

    val_df = df[
        df["id_planta"].isin(train_plants)
        & (df["timestamp"] >= val_start)
        & (df["timestamp"] <= val_end)
    ].copy()

    test_df = df[
        (df["id_planta"] == test_plant)
        & (df["timestamp"] >= test_start)
        & (df["timestamp"] <= test_end)
    ].copy()

    if train_df.empty:
        raise ValueError("El conjunto de train esta vacio.")
    if val_df.empty:
        raise ValueError("El conjunto de validacion esta vacio.")
    if test_df.empty:
        raise ValueError("El conjunto de test esta vacio.")

    logging.info(
        "Split: train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Limpieza basica
# ---------------------------------------------------------------------------
def basic_cleaning(
    df: pd.DataFrame,
    clip_power_lower: float = 0.0,
    clip_radiation_lower: float = 0.0,
) -> pd.DataFrame:
    """
    Limpieza conservadora:
    - Elimina filas donde 'power' es NaN.
    - Recorta 'power' y 'radiation' por debajo a 0.

    Parameters
    ----------
    df : pd.DataFrame
    clip_power_lower : float
        Valor minimo para 'power' (por defecto 0.0).
    clip_radiation_lower : float
        Valor minimo para 'radiation' (por defecto 0.0).
    """
    out = df.copy()
    before = len(out)
    out = out[out["power"].notna()].copy()
    removed = before - len(out)
    if removed > 0:
        logging.warning("Eliminadas %d filas con power=NaN.", removed)

    out["power"] = out["power"].clip(lower=clip_power_lower)
    out["radiation"] = out["radiation"].clip(lower=clip_radiation_lower)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Consistencia temporal
# ---------------------------------------------------------------------------
def check_time_consistency(
    df: pd.DataFrame,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Comprueba la consistencia temporal por planta calculando el numero de
    huecos respecto a la frecuencia esperada.

    Parameters
    ----------
    df : pd.DataFrame
    freq : str
        Frecuencia esperada entre registros (por defecto '15min').

    Returns
    -------
    pd.DataFrame con columnas: id_planta, n_rows, n_gaps, gap_ratio,
    most_common_diff.
    """
    df = df.sort_values(["id_planta", "timestamp"]).copy()
    expected = pd.to_timedelta(freq)
    results = []

    for plant_id, g in df.groupby("id_planta"):
        diffs = g["timestamp"].diff().dropna()
        n_rows = len(g)
        n_gaps = int((diffs != expected).sum())
        results.append({
            "id_planta": plant_id,
            "n_rows": n_rows,
            "n_gaps": n_gaps,
            "gap_ratio": float(n_gaps) / max(n_rows - 1, 1),
            "most_common_diff": diffs.value_counts().idxmax(),
        })

    return pd.DataFrame(results)
