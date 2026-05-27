#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/Preparacion_Dataset_15min_LECA1.py
-----------------------------------------
Consolida los archivos CSV de 15 minutos de la planta LECA1 en un
unico DataFrame, genera las variables temporales y guarda el resultado
en DATA/Datos_LECA1_15min.csv.

Estructura esperada en el directorio de entrada:
    <input_folder>/
        <anyo>/
            <mes>/
                *.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def crear_dataframe(input_folder: Path, output_path: Path) -> None:
    """
    Lee todos los CSV diarios de LECA1, los consolida y guarda el resultado.

    Parameters
    ----------
    input_folder : Path
        Directorio raiz que contiene las subcarpetas de anyo y mes.
    output_path : Path
        Ruta completa del CSV de salida.
    """
    dfs = []
    error_files = []

    for year_folder in sorted(input_folder.iterdir()):
        if not year_folder.is_dir():
            continue
        print(f"Procesando anyo: {year_folder.name}")

        for month_folder in sorted(year_folder.iterdir()):
            if not month_folder.is_dir():
                continue
            print(f"  Mes: {month_folder.name}")

            for file_path in sorted(month_folder.glob("*.csv")):
                try:
                    df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

                    # Seleccionar columnas: admite dos nombres posibles para la potencia
                    if "LECA 1" in df.columns:
                        df = df[["FECHA/HORA", "RADIACION SOLAR", "LECA 1"]]
                    elif "LECA1" in df.columns:
                        df = df[["FECHA/HORA", "RADIACION SOLAR", "LECA1"]]
                    else:
                        raise KeyError(
                            f"No se encontro columna de potencia ('LECA 1' o 'LECA1') "
                            f"en {file_path.name}. Columnas disponibles: {list(df.columns)}"
                        )

                    df.columns = ["timestamp", "radiation", "power"]

                    # Limpiar marcadores nulos del sistema origen
                    df = df.replace(r"\\N", pd.NA, regex=True)

                    df["radiation"] = pd.to_numeric(df["radiation"], errors="coerce")
                    df["power"] = pd.to_numeric(df["power"], errors="coerce")
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

                    dfs.append(df)

                except Exception as exc:
                    print(f"    Error en {file_path.name}: {exc}")
                    error_files.append(file_path.name)

    if not dfs:
        raise RuntimeError(
            "No se encontro ningun archivo valido en la carpeta de entrada. "
            "Revisa la ruta y la estructura de directorios."
        )

    # --- Consolidacion ---
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = (
        final_df
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    # --- Limpieza de valores fisicamente imposibles ---
    final_df["radiation"] = final_df["radiation"].fillna(0).clip(lower=0)
    final_df["power"] = final_df["power"].fillna(0).clip(lower=0)

    # --- Variables temporales ---
    print("Generando variables temporales...")
    ts = final_df["timestamp"]
    final_df["day_of_year"] = ts.dt.dayofyear
    final_df["hour"] = ts.dt.hour
    final_df["minute"] = ts.dt.minute
    final_df["month"] = ts.dt.month
    final_df["weekday"] = ts.dt.weekday
    final_df["is_weekend"] = (final_df["weekday"] >= 5).astype(int)

    # Codificacion ciclica
    final_df["sin_day"] = np.sin(2 * np.pi * final_df["day_of_year"] / 365)
    final_df["cos_day"] = np.cos(2 * np.pi * final_df["day_of_year"] / 365)
    final_df["sin_hour"] = np.sin(2 * np.pi * final_df["hour"] / 24)
    final_df["cos_hour"] = np.cos(2 * np.pi * final_df["hour"] / 24)

    # --- Guardado ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"\nDataFrame consolidado ({len(final_df)} registros) guardado en '{output_path}'")
    print(f"Rango temporal: {final_df['timestamp'].min()} -> {final_df['timestamp'].max()}")
    print(f"Columnas: {list(final_df.columns)}")

    if error_files:
        print(f"\nArchivos con errores ({len(error_files)}): {error_files}")


if __name__ == "__main__":
    # Rutas relativas a la raiz del repositorio.
    # Ejecutar desde: TFM-FV/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    INPUT_FOLDER = PROJECT_ROOT / "data" / "raw" / "LECA1"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "Datos_LECA1_15min.csv"

    if OUTPUT_PATH.exists():
        print(f"El archivo de salida ya existe: {OUTPUT_PATH}")
        print("Elimina el archivo manualmente si deseas regenerarlo.")
    else:
        crear_dataframe(INPUT_FOLDER, OUTPUT_PATH)
