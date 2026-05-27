#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/Analisis_datos_LECA1.py
------------------------------
Funciones de limpieza, validacion y analisis exploratorio del dataset
de produccion fotovoltaica de la planta LECA1.

Incluye:
- Carga y filtrado temporal
- Limpieza de nulos, negativos y dias vacios
- Correcciones fisicas (radiacion nocturna, potencia sin radiacion)
- Deteccion y relleno de huecos temporales
- Deteccion y correccion de outliers por IQR
- Analisis de correlacion mensual
- Generacion de graficos estaticos (Matplotlib) e interactivos (Plotly)

Uso como script independiente
------------------------------
    python utils/Analisis_datos_LECA1.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuracion (solo usada al ejecutar como script independiente)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

_ARCHIVO_ENTRADA = _DATA_DIR / "Datos_LECA1_Rellenos.csv"
_ARCHIVO_SALIDA_CSV = _DATA_DIR / "Datos_LECA1_Limpio.csv"
_ARCHIVO_SALIDA_HTML = _DATA_DIR / "Analisis_Interactivo_LECA1.html"

_FECHA_INICIO = "2022-01-01"
_FECHA_FIN = "2024-05-31"
_RELLENAR_HUECOS = True


# ---------------------------------------------------------------------------
# Carga y filtrado
# ---------------------------------------------------------------------------
def cargar_y_filtrar_datos(
    ruta: str | Path,
    fecha_inicio: str,
    fecha_fin: str,
    filtrar_datos: bool = True,
) -> pd.DataFrame:
    """
    Carga el CSV de la planta, convierte la columna timestamp y filtra por
    el rango de fechas indicado.

    Parameters
    ----------
    ruta : str o Path
        Ruta del archivo CSV de entrada.
    fecha_inicio, fecha_fin : str
        Rango temporal en formato 'YYYY-MM-DD'.
    filtrar_datos : bool
        Si True, recorta el DataFrame al rango indicado.

    Returns
    -------
    pd.DataFrame filtrado.

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    """
    ruta = Path(ruta)
    logging.info("Cargando datos desde: %s", ruta)

    try:
        df = pd.read_csv(ruta, parse_dates=["timestamp"])
    except FileNotFoundError:
        logging.error("Archivo no encontrado: %s", ruta)
        raise

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if filtrar_datos:
        mask = (df["timestamp"] >= fecha_inicio) & (df["timestamp"] <= fecha_fin)
        df_out = df.loc[mask].copy()
    else:
        df_out = df.copy()

    logging.info(
        "Registros totales: %d | Tras filtro temporal: %d", len(df), len(df_out)
    )
    return df_out


# ---------------------------------------------------------------------------
# Limpieza de nulos y negativos
# ---------------------------------------------------------------------------
def limpiar_nulos_y_negativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena NaN con 0 y recorta valores negativos en 'radiation' y 'power'.

    Los valores negativos en estas columnas carecen de sentido fisico y se
    tratan como errores de medicion.
    """
    logging.info("Limpiando nulos y valores negativos...")

    missing = df.isna().sum()
    if missing.sum() > 0:
        logging.warning("Valores NaN detectados:\n%s", missing[missing > 0].to_string())
        df["radiation"] = df["radiation"].fillna(0)
        df["power"] = df["power"].fillna(0)
    else:
        logging.info("No se encontraron valores NaN.")

    for col in ["radiation", "power"]:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            logging.warning("%d valores negativos en '%s' corregidos a 0.", n_neg, col)
            df[col] = df[col].clip(lower=0)
        else:
            logging.info("Sin valores negativos en '%s'.", col)

    return df


# ---------------------------------------------------------------------------
# Eliminacion de dias vacios
# ---------------------------------------------------------------------------
def eliminar_dias_vacios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los dias completos cuya suma de radiacion o produccion es cero.

    Genera un grafico de barras con los dias eliminados marcados en rojo
    si se detectan dias vacios en alguna de las dos columnas.
    """
    logging.info("Comprobando dias sin radiacion o produccion...")

    daily_rad = df.groupby(df["timestamp"].dt.date)["radiation"].sum()
    daily_pow = df.groupby(df["timestamp"].dt.date)["power"].sum()
    days_no_rad = daily_rad[daily_rad == 0]
    days_no_pow = daily_pow[daily_pow == 0]

    def _log_empty_days(col_name: str, days):
        if days.empty:
            logging.info("Todos los dias tienen algun registro de %s.", col_name)
        else:
            sample = ", ".join(str(d) for d in list(days.index[:5]))
            extra = f" (y {len(days) - 5} mas)" if len(days) > 5 else ""
            logging.warning(
                "%d dias sin %s: %s%s", len(days), col_name, sample, extra
            )

    _log_empty_days("radiacion", days_no_rad)
    _log_empty_days("produccion", days_no_pow)

    # Grafico solo si hay dias vacios
    if not days_no_rad.empty or not days_no_pow.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        ax1.bar(daily_rad.index, daily_rad, color="skyblue", label="Radiacion diaria")
        ax1.set_title(
            f"Suma diaria de radiacion (dias a eliminar: {len(days_no_rad)})"
        )
        ax1.set_ylabel("Radiacion acumulada")
        if not days_no_rad.empty:
            for day in days_no_rad.index:
                ax1.axvline(x=day, color="red", linestyle="-", alpha=0.5, linewidth=2)
            ax1.plot([], [], color="red", label="Dias sin datos", linewidth=2)
        ax1.legend()

        ax2.bar(daily_pow.index, daily_pow, color="orange", label="Produccion diaria")
        ax2.set_title(
            f"Suma diaria de produccion (dias a eliminar: {len(days_no_pow)})"
        )
        ax2.set_ylabel("Produccion acumulada")
        if not days_no_pow.empty:
            for day in days_no_pow.index:
                ax2.axvline(x=day, color="red", linestyle="-", alpha=0.5, linewidth=2)
            ax2.plot([], [], color="red", label="Dias sin datos", linewidth=2)
        ax2.legend()

        plt.xlabel("Fecha")
        plt.tight_layout()
        plt.show()

    # Eliminacion
    if not days_no_rad.empty:
        before = len(df)
        df = df[~df["timestamp"].dt.date.isin(days_no_rad.index)]
        logging.info(
            "Eliminadas %d filas por dias sin radiacion.", before - len(df)
        )

    if not days_no_pow.empty:
        before = len(df)
        df = df[~df["timestamp"].dt.date.isin(days_no_pow.index)]
        logging.info(
            "Eliminadas %d filas por dias sin produccion.", before - len(df)
        )

    return df


# ---------------------------------------------------------------------------
# Correccion de radiacion nocturna
# ---------------------------------------------------------------------------
def corregir_radiacion_nocturna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuerza a cero la radiacion registrada entre las 22:00 y las 06:00.

    Lecturas positivas en ese intervalo indican un error del sensor.
    Genera un grafico de los valores corregidos si se detectan anomalias.
    """
    logging.info("Comprobando radiacion nocturna (22:00-06:00)...")

    night = df[(df["hour"] < 6) | (df["hour"] > 22)]
    anomalos = night[night["radiation"] > 0]

    if anomalos.empty:
        logging.info("Sin radiacion nocturna detectada.")
    else:
        logging.warning(
            "%d registros con radiacion nocturna. Se corrigen a 0.", len(anomalos)
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            night["timestamp"], night["radiation"],
            color="red", linewidth=0.8, label="Radiacion nocturna",
        )
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Radiacion (W/m²)")
        ax.set_title("Radiacion registrada en horario nocturno")
        ax.grid(True, alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.show()

        df.loc[night.index, "radiation"] = 0

    return df


# ---------------------------------------------------------------------------
# Comprobacion de radiacion constante
# ---------------------------------------------------------------------------
def comprobar_radiacion_constante(df: pd.DataFrame) -> pd.DataFrame:
    """
    Emite una advertencia si existen dias con radiacion constante y positiva
    durante las 24 horas (sintoma de sensor bloqueado).

    No modifica los datos: la correccion se delega en fill_radiation_data_LECA1.
    """
    logging.info("Comprobando dias con radiacion constante (sensor bloqueado)...")

    daily = df.groupby(df["timestamp"].dt.date)["radiation"].agg(["nunique", "mean"])
    constant = daily[(daily["nunique"] == 1) & (daily["mean"] > 0)]

    if constant.empty:
        logging.info("No se detectaron dias con radiacion constante.")
    else:
        logging.warning(
            "%d dias con radiacion constante detectados:", len(constant)
        )
        for day, stats in constant.iterrows():
            logging.warning("  - %s: valor = %.2f", day, stats["mean"])

    return df


# ---------------------------------------------------------------------------
# Correccion de potencia sin radiacion
# ---------------------------------------------------------------------------
def corregir_potencia_sin_radiacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuerza a cero la produccion registrada en instantes donde la radiacion es 0.

    Registros con produccion y radiacion nula simultaneamente son fisicamente
    inconsistentes y se tratan como errores de medicion.
    """
    logging.info("Comprobando produccion sin radiacion...")

    mask = (df["radiation"] == 0) & (df["power"] > 0)
    n = mask.sum()

    if n == 0:
        logging.info("Sin registros de produccion con radiacion nula.")
    else:
        dias = df.loc[mask, "timestamp"].dt.date.unique()
        logging.warning(
            "%d registros con produccion y radiacion = 0 en %d dias. Se corrigen a 0.",
            n, len(dias),
        )
        df.loc[mask, "power"] = 0

    return df


# ---------------------------------------------------------------------------
# Huecos temporales
# ---------------------------------------------------------------------------
def procesar_huecos_temporales(
    df: pd.DataFrame, rellenar: bool = True
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Reindexa el DataFrame a frecuencia estricta de 15 minutos y rellena los
    huecos detectados en dos niveles:

    1. Interpolacion lineal para huecos de hasta 2 horas (8 periodos).
    2. Copia del mismo intervalo del dia anterior para huecos mayores
       (hasta 3 dias de lookback).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'timestamp'.
    rellenar : bool
        Si False, reindexa pero no rellena.

    Returns
    -------
    df_relleno : DataFrame con frecuencia completa de 15 minutos.
    huecos_indices : DatetimeIndex con los timestamps que eran huecos.
    """
    logging.info("Detectando y tratando huecos temporales...")

    df_sorted = df.sort_values("timestamp").set_index("timestamp")
    rango = pd.date_range(
        start=df_sorted.index.min(),
        end=df_sorted.index.max(),
        freq="15min",
    )
    huecos_indices = rango.difference(df_sorted.index)

    if len(huecos_indices) > 0:
        logging.info(
            "%d intervalos de 15min faltantes. Generando grafico de distribucion...",
            len(huecos_indices),
        )
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.vlines(
            huecos_indices, ymin=0, ymax=1,
            colors="red", linewidth=0.8, alpha=0.8, label="Dato faltante",
        )
        ax.set_title(
            f"Distribucion temporal de huecos ({len(huecos_indices)} intervalos de 15 min)"
        )
        ax.set_xlabel("Fecha")
        ax.set_yticks([])
        ax.set_xlim(rango.min(), rango.max())
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        logging.info("No se detectaron huecos temporales.")

    df_relleno = df_sorted.reindex(rango)

    if not rellenar:
        logging.info("Relleno de huecos desactivado (rellenar=False).")
        return (
            df_relleno.reset_index().rename(columns={"index": "timestamp"}),
            huecos_indices,
        )

    cols = ["radiation", "power"]

    # Nivel 1: interpolacion para huecos cortos (<2h)
    df_relleno[cols] = df_relleno[cols].interpolate(method="time", limit=8)
    logging.info("Huecos cortos (<2h) interpolados.")

    # Nivel 2: copia del dia anterior para huecos medianos
    periodos_dia = 96  # 24h * 4 periodos/h
    for _ in range(3):
        for col in cols:
            df_relleno[col] = df_relleno[col].fillna(df_relleno[col].shift(periodos_dia))

    nans_restantes = df_relleno[cols].isna().sum().sum()
    if nans_restantes > 0:
        logging.warning(
            "%d valores NaN no resueltos (huecos superiores a 3 dias).",
            nans_restantes,
        )
    else:
        logging.info("Todos los huecos cubiertos.")

    # Restaurar columnas auxiliares derivadas del indice
    df_relleno["Mes"] = df_relleno.index.to_period("M").astype(str)
    df_relleno["hour"] = df_relleno.index.hour

    return (
        df_relleno.reset_index().rename(columns={"index": "timestamp"}),
        huecos_indices,
    )


# ---------------------------------------------------------------------------
# Correlacion mensual
# ---------------------------------------------------------------------------
def analizar_correlacion_mensual(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Calcula la correlacion mensual entre radiacion y produccion y genera un
    grafico de barras resaltando los meses con correlacion inferior a 0.85.

    Returns
    -------
    df_corr : DataFrame con columnas 'Mes' y 'Correlacion'.
    fig : Figure de Matplotlib.
    """
    logging.info("Calculando correlacion mensual radiacion-produccion...")

    df_local = df.dropna(subset=["radiation", "power"]).copy()
    df_local["Mes_Periodo"] = df_local["timestamp"].dt.to_period("M")

    correlations = df_local.groupby("Mes_Periodo")[["radiation", "power"]].apply(
        lambda x: x["radiation"].corr(x["power"])
    )
    df_corr = correlations.reset_index(name="Correlacion")
    df_corr["Mes"] = df_corr["Mes_Periodo"].astype(str)

    baja_corr = df_corr[df_corr["Correlacion"] < 0.85]
    if not baja_corr.empty:
        logging.warning(
            "Meses con correlacion baja (<0.85):\n%s",
            baja_corr[["Mes", "Correlacion"]].to_string(index=False),
        )
    else:
        logging.info("Correlacion superior a 0.85 en todos los meses.")

    colors = ["red" if c < 0.85 else "steelblue" for c in df_corr["Correlacion"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_corr["Mes"], df_corr["Correlacion"], color=colors)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Correlacion")
    ax.set_title("Correlacion mensual: Radiacion vs Produccion")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    return df_corr, fig


# ---------------------------------------------------------------------------
# Deteccion y correccion de outliers
# ---------------------------------------------------------------------------
def detectar_y_corregir_outliers(
    df: pd.DataFrame, iqr_multiplier: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta outliers en 'power' y 'radiation' mediante el metodo IQR y los
    corrige por clipping.

    Parameters
    ----------
    df : pd.DataFrame
    iqr_multiplier : float
        Multiplicador del IQR para definir los limites (por defecto 3.0).

    Returns
    -------
    df : DataFrame corregido.
    outliers_power : DataFrame con los registros outliers de 'power'.
    outliers_rad : DataFrame con los registros outliers de 'radiation'.
    """
    logging.info("Detectando outliers (IQR x %.1f)...", iqr_multiplier)
    outliers = {}

    for col in ["power", "radiation"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        outliers[col] = df[mask].copy()

        if not outliers[col].empty:
            logging.warning(
                "%s: %d outliers detectados. Corregidos por clipping [%.2f, %.2f].",
                col, len(outliers[col]), lower, upper,
            )
            df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            logging.info("%s: sin outliers extremos.", col)

    return df, outliers["power"], outliers["radiation"]


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def graficar_matplotlib_estatico(
    df: pd.DataFrame, huecos_indices: pd.DatetimeIndex
) -> None:
    """
    Genera un grafico estatico de Matplotlib con la serie temporal completa
    de radiacion y produccion, marcando los huecos detectados.
    """
    logging.info("Generando grafico estatico...")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(
        df["timestamp"], df["radiation"],
        label="Radiacion Solar", color="steelblue", linewidth=0.5, alpha=0.7,
    )
    ax.plot(
        df["timestamp"], df["power"],
        label="Produccion Solar", color="darkorange", linewidth=0.5, alpha=0.7,
    )

    if len(huecos_indices) > 0:
        y_vals = np.full(len(huecos_indices), -50)
        ax.scatter(
            huecos_indices, y_vals,
            color="red", marker="x", label="Huecos detectados/rellenados", s=20,
        )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.set_title("Radiacion y produccion a lo largo del tiempo")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    plt.show()


def generar_reporte_interactivo(
    df: pd.DataFrame,
    outliers_power: pd.DataFrame,
    outliers_rad: pd.DataFrame,
    huecos_indices: pd.DatetimeIndex,
    archivo_salida: str | Path,
) -> go.Figure:
    """
    Genera un grafico interactivo Plotly con radiacion, produccion, temperatura
    (si esta disponible), outliers y huecos, y lo exporta como HTML.

    Parameters
    ----------
    df : DataFrame con los datos limpios.
    outliers_power : DataFrame con outliers de 'power'.
    outliers_rad : DataFrame con outliers de 'radiation'.
    huecos_indices : DatetimeIndex con los timestamps de los huecos.
    archivo_salida : Ruta del HTML de salida.

    Returns
    -------
    go.Figure
    """
    logging.info("Generando reporte interactivo: %s", archivo_salida)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["radiation"],
        mode="lines", name="Radiacion",
        line=dict(color="steelblue", width=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["power"],
        mode="lines", name="Produccion",
        line=dict(color="darkorange", width=0.8),
    ))

    if "T_ambiente" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["T_ambiente"],
            mode="lines", name="Temperatura",
            line=dict(color="seagreen", width=0.8),
        ))

    if not outliers_power.empty:
        fig.add_trace(go.Scatter(
            x=outliers_power["timestamp"], y=outliers_power["power"],
            mode="markers", name="Outliers (power)",
            marker=dict(color="purple", symbol="circle-open", size=8),
        ))

    if not outliers_rad.empty:
        fig.add_trace(go.Scatter(
            x=outliers_rad["timestamp"], y=outliers_rad["radiation"],
            mode="markers", name="Outliers (radiation)",
            marker=dict(color="red", symbol="diamond-open", size=8),
        ))

    if len(huecos_indices) > 0:
        fig.add_trace(go.Scatter(
            x=huecos_indices, y=[0] * len(huecos_indices),
            mode="markers", name="Huecos originales",
            marker=dict(color="black", symbol="x", size=6),
            hoverinfo="x",
        ))

    fig.update_layout(
        title="Analisis interactivo — Planta fotovoltaica LECA1",
        xaxis_title="Fecha",
        yaxis_title="W / W/m²",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        xaxis_rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )

    Path(archivo_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(archivo_salida))
    logging.info("Reporte guardado en: %s", archivo_salida)

    return fig


# ---------------------------------------------------------------------------
# Script independiente
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta el pipeline completo de limpieza y analisis."""
    logging.info("Inicio del pipeline de analisis fotovoltaico.")

    df = cargar_y_filtrar_datos(_ARCHIVO_ENTRADA, _FECHA_INICIO, _FECHA_FIN)
    df = limpiar_nulos_y_negativos(df)
    df = eliminar_dias_vacios(df)
    df = corregir_radiacion_nocturna(df)
    df = comprobar_radiacion_constante(df)
    df = corregir_potencia_sin_radiacion(df)

    df, huecos_indices = procesar_huecos_temporales(df, rellenar=_RELLENAR_HUECOS)

    df_corr, fig_corr = analizar_correlacion_mensual(df)
    fig_corr.show()

    df, outliers_pow, outliers_rad = detectar_y_corregir_outliers(df)

    graficar_matplotlib_estatico(df, huecos_indices)
    generar_reporte_interactivo(
        df, outliers_pow, outliers_rad, huecos_indices, _ARCHIVO_SALIDA_HTML
    )

    df.to_csv(_ARCHIVO_SALIDA_CSV, index=False)
    logging.info("Dataset final guardado en: %s", _ARCHIVO_SALIDA_CSV)
    logging.info("Pipeline completado.")


if __name__ == "__main__":
    main()
