#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/models.py
--------------
Definicion de los modelos de regresion utilizados en la comparativa del TFM.

Incluye un conjunto de modelos base que cubre el espectro desde regresion
lineal hasta gradient boosting y redes neuronales, facilitando la comparacion
sistematica de su rendimiento en la tarea de prediccion de produccion
fotovoltaica normalizada (power_pu).

Modelos incluidos
-----------------
- Ridge          : regresion lineal regularizada (baseline lineal).
- Random Forest  : ensemble de arboles de decision.
- Extra Trees    : ensemble de arboles extremadamente aleatorios.
- XGBoost        : gradient boosting con regularizacion L1/L2.
- LightGBM       : gradient boosting orientado a eficiencia computacional.
- CatBoost       : gradient boosting robusto sin preprocesado de categoricas.
- MLP            : perceptron multicapa con escalado previo (Pipeline).

Nota sobre el MLP
-----------------
Es el unico modelo que requiere escalado de features. Se envuelve en un
sklearn Pipeline con StandardScaler para garantizar que las features esten
normalizadas antes de la propagacion hacia adelante.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    """
    Devuelve un diccionario con los modelos de la comparativa base.

    Todos los modelos estan preconfigurados con hiperparametros razonables
    para datos de series temporales tabulares de 15 minutos. No requieren
    ajuste adicional para la evaluacion inicial; 

    Parameters
    ----------
    random_state : int
        Semilla de aleatoriedad para reproducibilidad (por defecto 42).

    Returns
    -------
    dict {nombre_modelo: instancia_inicializada}
    """
    models: Dict[str, object] = {}

    # Baseline lineal regularizado
    models["Ridge"] = Ridge(alpha=1.0)

    # Ensemble de arboles no lineal
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    # Ensemble de arboles extremadamente aleatorios (mas rapido que RF)
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=random_state,
    )

    # Gradient boosting con regularizacion explicita
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    # Gradient boosting orientado a eficiencia (leaf-wise)
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1,
    )

    # Gradient boosting robusto con manejo nativo de categoricas
    models["CatBoost"] = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )

    # Red neuronal multicapa con escalado previo obligatorio
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )),
    ])

    return models
