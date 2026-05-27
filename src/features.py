#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/features.py
----------------
Pipeline de ingenieria de variables para el modelo de prediccion de
produccion fotovoltaica multi-planta.

Incluye:
- Variables temporales ciclicas (seno/coseno de hora y dia del año)
- Bandera de horas de luz
- Interacciones fisicas (radiacion x temperatura, radiacion^2)
- Variables de regimen de irradiancia
- Lags y medias moviles por planta (sin fuga de informacion)
- Deltas de potencia y radiacion
- Codificacion one-hot de planta

Nota sobre fuga de informacion
-------------------------------
Los lags y rolling features aplican shift(1) antes del calculo, de modo
que cada fila solo usa informacion del pasado estricto. Los lags se calculan
agrupando por planta para no mezclar el final de una serie con el inicio de
la siguiente.
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
    """
    Parametros del pipeline de ingenieria de variables.

    Attributes
    ----------
    lag_steps_power : tuple
        Pasos de lag para el target (en periodos de 15 min).
    lag_steps_radiation : tuple
        Pasos de lag para la radiacion.
    rolling_windows_power : tuple
        Ventanas para la media movil del target.
    rolling_windows_radiation : tuple
        Ventanas para la media movil de la radiacion.
    add_interactions : bool
        Si True, añade interacciones fisicas (radiation*T, radiation^2).
    add_daylight_flag : bool
        Si True, añade bandera binaria de horas de luz.
    drop_na_after_features : bool
        Si True, elimina filas con NaN generados por lags/rolling.
    one_hot_encode_plant : bool
        Si True, añade columnas one-hot del identificador de planta.
    """
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
    """
    Genera variables temporales basicas y sus codificaciones ciclicas desde
    la columna timestamp.

    Las codificaciones ciclicas (seno/coseno) permiten al modelo captar la
    periodicidad sin discontinuidades en los extremos del ciclo.
    """
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
    """
    Añade una bandera binaria (is_daylight) que indica si el registro
    corresponde a un periodo con irradiancia suficiente para produccion.

    Parameters
    ----------
    radiation_threshold : float
        Umbral de radiacion en W/m² por encima del cual se considera luz
        diurna (por defecto 20.0).
    """
    out = df.copy()
    out["is_daylight"] = (out["radiation"].fillna(0) > radiation_threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# Interacciones fisicas
# ---------------------------------------------------------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade interacciones fisicas basicas:
    - radiation_x_temp: producto de radiacion y temperatura.
    - radiation_sq: cuadrado de la radiacion (relacion no lineal
      entre irradiancia y potencia).
    """
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
    """
    Crea variables de lag para el target y la radiacion, agrupando por planta
    para evitar que el final de una planta contamine el inicio de otra.

    Parameters
    ----------
    group_col : str
        Columna de agrupacion (normalmente PLANT_COLUMN).
    target_col : str
        Columna objetivo sobre la que calcular los lags.
    radiation_col : str
        Columna de radiacion sobre la que calcular los lags.
    lag_steps_power : iterable de int
        Pasos de lag para el target.
    lag_steps_radiation : iterable de int
        Pasos de lag para la radiacion.
    """
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
    Crea medias y desviaciones tipicas moviles por planta usando
    exclusivamente informacion del pasado (shift(1) antes del rolling).

    El shift previo garantiza que la ventana de la fila t solo incluye
    valores hasta t-1, evitando fuga de informacion.
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
    Calcula el cambio entre lags consecutivos para captar transiciones rapidas
    en la produccion y la radiacion.

    Solo se generan si los lags correspondientes ya existen en el DataFrame.
    Usa exclusivamente informacion del pasado (lag_1 - lag_2).
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
    """
    Clasifica cada registro en un regimen de irradiancia:
    - noche: radiacion < 10 W/m²
    - baja:  10 <= radiacion < 200 W/m²
    - media: 200 <= radiacion < 600 W/m²
    - alta:  radiacion >= 600 W/m²
    """
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
# One-hot de planta
# ---------------------------------------------------------------------------
def encode_plant_id(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """
    Codifica el identificador de planta mediante one-hot encoding.

    Parameters
    ----------
    drop_first : bool
        Si True, elimina la primera categoria para evitar multicolinealidad
        perfecta (recomendado para modelos lineales).
    """
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
    Ejecuta el pipeline completo de ingenieria de variables en el orden:

    1. Variables temporales ciclicas
    2. Bandera de luz diurna
    3. Interacciones fisicas
    4. Clasificacion de regimen de irradiancia (one-hot)
    5. Lags de target y radiacion por planta
    6. Medias y desviaciones tipicas moviles por planta
    7. Deltas de potencia y radiacion
    8. One-hot del identificador de planta
    9. Eliminacion de NaN generados por lags/rolling

    Parameters
    ----------
    df : pd.DataFrame
        Dataset unificado con columnas obligatorias validadas.
    config : FeatureConfig
        Configuracion del pipeline.

    Returns
    -------
    pd.DataFrame con todas las variables generadas.
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
    Devuelve las columnas de entrada al modelo, excluyendo el target,
    identificadores y columnas que introducen fuga de informacion.

    Columnas excluidas:
    - power_pu (target)
    - timestamp, id_planta (identificadores)
    - power (version cruda del target, fuga directa)
    - p_nominal_kw (auxiliar de normalizacion)
    - Mes, season (redundantes con variables temporales ciclicas)
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
    return [c for c in df.columns if c not in exclude]
