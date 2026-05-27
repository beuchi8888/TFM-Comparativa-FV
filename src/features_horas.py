#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/features_horas.py
----------------------
Extension de src/features.py con soporte para prediccion multi-horizonte.

Añade las funciones:
- add_forecast_horizon_targets: genera columnas target desplazadas hacia
  adelante para distintos horizontes de prediccion (t+1, t+4, t+16...).
- get_horizon_target_columns: devuelve los nombres de dichas columnas.

El resto de funciones son identicas a src/features.py.
Importar este modulo cuando se trabajen con modelos de horizonte multiple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "radiation",
    "T_ambiente",
]

TEMPORAL_FEATURES = [
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
]

TARGET_COLUMN = "power_pu"
TIMESTAMP_COLUMN = "timestamp"
PLANT_COLUMN = "id_planta"


# ---------------------------------------------------------------------------
# Configuracion del pipeline
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Parametros del pipeline de ingenieria de variables."""
    lag_steps_power: tuple[int, ...] = (1, 2, 4)
    lag_steps_radiation: tuple[int, ...] = (1, 2)
    rolling_windows_power: tuple[int, ...] = (4, 8)
    rolling_windows_radiation: tuple[int, ...] = (4,)
    add_interactions: bool = True
    add_daylight_flag: bool = True
    drop_na_after_features: bool = True
    one_hot_encode_plant: bool = True


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------
def _check_required_columns(df: pd.DataFrame) -> None:
    required = {TIMESTAMP_COLUMN, PLANT_COLUMN, "radiation", "T_ambiente", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas necesarias para crear features: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Variables temporales
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables temporales basicas y sus codificaciones ciclicas."""
    out = df.copy()

    if not np.issubdtype(out[TIMESTAMP_COLUMN].dtype, np.datetime64):
        out[TIMESTAMP_COLUMN] = pd.to_datetime(out[TIMESTAMP_COLUMN], errors="coerce")

    ts = out[TIMESTAMP_COLUMN]
    out["day_of_year"] = ts.dt.dayofyear
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["month"] = ts.dt.month
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    out["sin_day"] = np.sin(2 * np.pi * out["day_of_year"] / 365.0)
    out["cos_day"] = np.cos(2 * np.pi * out["day_of_year"] / 365.0)

    hour_float = out["hour"] + out["minute"] / 60.0
    out["sin_hour"] = np.sin(2 * np.pi * hour_float / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hour_float / 24.0)

    return out


# ---------------------------------------------------------------------------
# Bandera de luz diurna
# ---------------------------------------------------------------------------
def add_daylight_feature(
    df: pd.DataFrame,
    radiation_threshold: float = 20.0,
) -> pd.DataFrame:
    """Añade bandera binaria de horas con irradiancia suficiente para produccion."""
    out = df.copy()
    out["is_daylight"] = (out["radiation"].fillna(0) > radiation_threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# Interacciones fisicas
# ---------------------------------------------------------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade radiation_x_temp y radiation_sq como interacciones fisicas basicas."""
    out = df.copy()
    out["radiation_x_temp"] = out["radiation"] * out["T_ambiente"]
    out["radiation_sq"] = out["radiation"] ** 2
    return out


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    radiation_col: str = "radiation",
    lag_steps_power: Iterable[int] = (1, 2, 4),
    lag_steps_radiation: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Crea lags de target y radiacion agrupando por planta."""
    out = df.copy()
    out = out.sort_values([group_col, TIMESTAMP_COLUMN]).reset_index(drop=True)
    grouped = out.groupby(group_col, group_keys=False)

    for lag in lag_steps_power:
        out[f"{target_col}_lag_{lag}"] = grouped[target_col].shift(lag)

    for lag in lag_steps_radiation:
        out[f"{radiation_col}_lag_{lag}"] = grouped[radiation_col].shift(lag)

    return out


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    radiation_col: str = "radiation",
    rolling_windows_power: Iterable[int] = (4, 8),
    rolling_windows_radiation: Iterable[int] = (4,),
) -> pd.DataFrame:
    """
    Medias y desviaciones tipicas moviles por planta usando shift(1)
    para evitar fuga de informacion.
    """
    out = df.copy()
    out = out.sort_values([group_col, TIMESTAMP_COLUMN]).reset_index(drop=True)
    grouped = out.groupby(group_col, group_keys=False)

    past_target = grouped[target_col].shift(1)
    past_radiation = grouped[radiation_col].shift(1)

    for window in rolling_windows_power:
        out[f"{target_col}_roll_mean_{window}"] = (
            past_target.groupby(out[group_col])
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{target_col}_roll_std_{window}"] = (
            past_target.groupby(out[group_col])
            .rolling(window=window, min_periods=2).std()
            .reset_index(level=0, drop=True)
        )

    for window in rolling_windows_radiation:
        out[f"{radiation_col}_roll_mean_{window}"] = (
            past_radiation.groupby(out[group_col])
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{radiation_col}_roll_std_{window}"] = (
            past_radiation.groupby(out[group_col])
            .rolling(window=window, min_periods=2).std()
            .reset_index(level=0, drop=True)
        )

    return out


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------
def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cambio entre lags consecutivos de potencia y radiacion.
    Solo se calcula si los lags correspondientes ya existen.
    """
    out = df.copy()

    if (
        f"{TARGET_COLUMN}_lag_1" in out.columns
        and f"{TARGET_COLUMN}_lag_2" in out.columns
    ):
        out[f"delta_{TARGET_COLUMN}_1"] = (
            out[f"{TARGET_COLUMN}_lag_1"] - out[f"{TARGET_COLUMN}_lag_2"]
        )

    if "radiation_lag_1" in out.columns and "radiation_lag_2" in out.columns:
        out["delta_radiation_1"] = out["radiation_lag_1"] - out["radiation_lag_2"]

    return out


# ---------------------------------------------------------------------------
# Regimen de irradiancia
# ---------------------------------------------------------------------------
def add_regime_class(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada registro en noche / baja / media / alta irradiancia."""
    def _classify(rad: float) -> str:
        if rad < 10:
            return "noche"
        if rad < 200:
            return "baja"
        if rad < 600:
            return "media"
        return "alta"

    out = df.copy()
    out["regime_class"] = out["radiation"].apply(_classify)
    return out


# ---------------------------------------------------------------------------
# Targets de horizonte multiple (exclusivo de este modulo)
# ---------------------------------------------------------------------------
def add_forecast_horizon_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 4, 16),
    group_col: str = PLANT_COLUMN,
) -> pd.DataFrame:
    """
    Crea columnas target desplazadas hacia adelante para distintos horizontes
    de prediccion, agrupando por planta.

    Cada horizonte h genera la columna power_pu_hN donde N es el numero de
    pasos de 15 minutos:

        h=1  -> power_pu_h1  -> t+15 min
        h=4  -> power_pu_h4  -> t+1 hora
        h=16 -> power_pu_h16 -> t+4 horas

    Los features de entrada (lags del pasado) son validos para cualquier
    horizonte. Las filas al final de cada planta tendran NaN en los targets
    de horizonte largo y deben eliminarse antes del entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
    horizons : tuple de int
        Horizontes de prediccion en pasos de 15 minutos.
    group_col : str
        Columna de agrupacion por planta.

    Returns
    -------
    pd.DataFrame con columnas adicionales power_pu_hN.
    """
    out = df.copy()
    out = out.sort_values([group_col, TIMESTAMP_COLUMN]).reset_index(drop=True)

    for h in horizons:
        col_name = f"{TARGET_COLUMN}_h{h}"
        out[col_name] = out.groupby(group_col)[TARGET_COLUMN].shift(-h)

    return out


def get_horizon_target_columns(
    horizons: tuple[int, ...] = (1, 4, 16)
) -> list[str]:
    """
    Devuelve los nombres de las columnas target de horizonte generadas por
    add_forecast_horizon_targets.
    """
    return [f"{TARGET_COLUMN}_h{h}" for h in horizons]


# ---------------------------------------------------------------------------
# One-hot de planta
# ---------------------------------------------------------------------------
def encode_plant_id(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encoding del identificador de planta."""
    out = df.copy()
    dummies = pd.get_dummies(
        out[PLANT_COLUMN], prefix="plant", drop_first=drop_first, dtype=int
    )
    return pd.concat([out, dummies], axis=1)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Pipeline completo de ingenieria de variables.
    Ver src/features.py para la descripcion detallada de cada paso.
    """
    _check_required_columns(df)

    out = df.copy()
    out = out.sort_values([PLANT_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    out = add_temporal_features(out)

    if config.add_daylight_flag:
        out = add_daylight_feature(out)

    if config.add_interactions:
        out = add_interaction_features(out)

    out = add_regime_class(out)
    out = pd.get_dummies(out, columns=["regime_class"], prefix="regime", dtype=int)

    out = add_lag_features(
        out,
        group_col=PLANT_COLUMN,
        target_col=TARGET_COLUMN,
        radiation_col="radiation",
        lag_steps_power=config.lag_steps_power,
        lag_steps_radiation=config.lag_steps_radiation,
    )

    out = add_rolling_features(
        out,
        group_col=PLANT_COLUMN,
        target_col=TARGET_COLUMN,
        radiation_col="radiation",
        rolling_windows_power=config.rolling_windows_power,
        rolling_windows_radiation=config.rolling_windows_radiation,
    )

    out = add_delta_features(out)

    if config.one_hot_encode_plant:
        out = encode_plant_id(out, drop_first=False)

    if config.drop_na_after_features:
        out = out.dropna().reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Seleccion de columnas de entrada
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas de entrada al modelo, excluyendo target,
    identificadores, columnas de fuga y columnas de horizonte (power_pu_hN).
    """
    exclude = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        "power",
        "p_nominal_kw",
        PLANT_COLUMN,
        "Mes",
        "season",
    }

    return [
        c for c in df.columns
        if c not in exclude
        and not (
            c.startswith(f"{TARGET_COLUMN}_h")
            and c[len(TARGET_COLUMN) + 2:].isdigit()
        )
    ]
