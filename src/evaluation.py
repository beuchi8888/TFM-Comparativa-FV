#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/evaluation.py
------------------
Funciones de evaluacion de modelos de prediccion fotovoltaica.

Incluye:
- Calculo de metricas estandar (MAE, RMSE, R²)
- Entrenamiento y evaluacion comparativa de multiples modelos
- Evaluacion restringida a horas de luz (registros con produccion > 0)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """
    Calcula MAE, RMSE y R² sobre el conjunto completo.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Returns
    -------
    dict con claves 'MAE', 'RMSE', 'R2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion comparativa
# ---------------------------------------------------------------------------
def train_and_evaluate_models(
    models: dict,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Entrena todos los modelos del diccionario y devuelve las metricas de
    validacion y test por separado, junto con los modelos entrenados.

    Parameters
    ----------
    models : dict
        Diccionario {nombre: instancia_de_modelo}.
    X_train, y_train : datos de entrenamiento.
    X_val, y_val : datos de validacion.
    X_test, y_test : datos de test.

    Returns
    -------
    df_val : DataFrame con metricas de validacion, ordenado por RMSE.
    df_test : DataFrame con metricas de test, ordenado por RMSE.
    trained_models : dict con los modelos ya entrenados.
    """
    results_val = []
    results_test = []
    trained_models = {}

    for name, model in models.items():
        logging.info("Entrenando: %s", name)
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred_val = model.predict(X_val)
        metrics_val = compute_metrics(y_val, y_pred_val)
        metrics_val["model"] = name
        results_val.append(metrics_val)

        y_pred_test = model.predict(X_test)
        metrics_test = compute_metrics(y_test, y_pred_test)
        metrics_test["model"] = name
        results_test.append(metrics_test)

        logging.info(
            "  Val  -> MAE: %.4f | RMSE: %.4f | R2: %.4f",
            metrics_val["MAE"], metrics_val["RMSE"], metrics_val["R2"],
        )
        logging.info(
            "  Test -> MAE: %.4f | RMSE: %.4f | R2: %.4f",
            metrics_test["MAE"], metrics_test["RMSE"], metrics_test["R2"],
        )

    df_val = pd.DataFrame(results_val).sort_values("RMSE").reset_index(drop=True)
    df_test = pd.DataFrame(results_test).sort_values("RMSE").reset_index(drop=True)

    return df_val, df_test, trained_models


# ---------------------------------------------------------------------------
# Metricas en horas de luz
# ---------------------------------------------------------------------------
def compute_metrics_daylight(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """
    Calcula metricas unicamente sobre los registros con produccion solar real,
    filtrando los periodos nocturnos donde la produccion es cero.

    Parameters
    ----------
    y_true : array-like
        Valores reales del target.
    y_pred : array-like
        Valores predichos.
    threshold : float
        Umbral de produccion por encima del cual se considera hora de luz
        (por defecto 0.0, es decir, cualquier produccion positiva).

    Returns
    -------
    dict con claves 'MAE', 'RMSE', 'R2'.

    Raises
    ------
    ValueError si no hay registros con produccion por encima del umbral.
    """
    mask = np.asarray(y_true) > threshold
    if mask.sum() == 0:
        raise ValueError(
            f"No hay registros con produccion > {threshold} para calcular metricas."
        )

    return compute_metrics(
        np.asarray(y_true)[mask],
        np.asarray(y_pred)[mask],
    )


def evaluate_all_models_daylight(
    trained_models: dict,
    X_test,
    y_test,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Genera una tabla comparativa de metricas en horas de luz para todos los
    modelos entrenados.

    Solo evalua los registros donde la produccion real supera el umbral
    indicado, lo que permite medir la precision del modelo durante los
    periodos en que la planta esta generando energia.

    Parameters
    ----------
    trained_models : dict
        Diccionario {nombre: modelo_entrenado}.
    X_test : features de test.
    y_test : target de test.
    threshold : float
        Umbral de produccion para filtrar horas de luz.

    Returns
    -------
    pd.DataFrame con metricas por modelo, ordenado por RMSE.
    """
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics_daylight(y_test, y_pred, threshold)
        metrics["model"] = name
        results.append(metrics)

    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
