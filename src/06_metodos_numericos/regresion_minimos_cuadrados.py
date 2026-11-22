"""Ajuste OLS de ventas globales frente a la nota de crítica.

Es un análisis histórico posterior al lanzamiento; no forma parte del
predictor de éxito pre-lanzamiento.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_PROCESADOS = RAIZ / "datos_procesados"
RUTA_DATASET = CARPETA_PROCESADOS / "videojuegos_integrado.csv"
RUTA_MODELO = CARPETA_PROCESADOS / "regresion_minimos_cuadrados.joblib"
RUTA_METRICAS = CARPETA_PROCESADOS / "metricas_regresion_minimos_cuadrados.json"
RUTA_PREDICCIONES = CARPETA_PROCESADOS / "predicciones_regresion_critica_ventas.csv"
RUTA_GRAFICO = CARPETA_PROCESADOS / "regresion_minimos_cuadrados.png"

ANIO_INICIO_TEST = 2012


def coeficientes_minimos_cuadrados(x, y):
    """Calcula pendiente e intercepto con las ecuaciones de las diapositivas."""
    n = len(x)
    if n < 3 or len(y) != n:
        raise ValueError("Se requieren al menos 3 pares (x, y) del mismo tamaño.")
    suma_x = float(np.sum(x))
    suma_y = float(np.sum(y))
    suma_x2 = float(np.sum(x ** 2))
    suma_xy = float(np.sum(x * y))
    denominador = n * suma_x2 - suma_x ** 2
    if np.isclose(denominador, 0):
        raise ValueError("No se puede estimar una pendiente si todos los x son iguales.")
    pendiente = (n * suma_xy - suma_x * suma_y) / denominador
    intercepto = float(np.mean(y) - pendiente * np.mean(x))
    return intercepto, float(pendiente)


def main():
    print("=" * 72)
    print("MÉTODOS NUMÉRICOS · REGRESIÓN LINEAL POR MÍNIMOS CUADRADOS")
    print("=" * 72)
    if not RUTA_DATASET.exists():
        raise SystemExit(
            f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
            "Ejecuta primero: python src/02_preparacion/merged.py"
        )

    df = pd.read_csv(
        RUTA_DATASET,
        usecols=["year", "critic_score", "global_sales"],
    )
    for columna in ("year", "critic_score", "global_sales"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=["year", "critic_score", "global_sales"]).copy()

    train_mask = df["year"] < ANIO_INICIO_TEST
    test_mask = df["year"] >= ANIO_INICIO_TEST
    train = df.loc[train_mask]
    test = df.loc[test_mask]
    if len(train) < 3 or test.empty or train["critic_score"].nunique() < 2:
        raise SystemExit("[ERROR] No hay suficientes datos para el ajuste temporal.")

    x_train = train["critic_score"].to_numpy(dtype=float)
    y_train = train["global_sales"].to_numpy(dtype=float)
    intercepto, pendiente = coeficientes_minimos_cuadrados(x_train, y_train)

    modelo = LinearRegression()
    modelo.fit(x_train.reshape(-1, 1), y_train)
    if not (
        np.isclose(modelo.intercept_, intercepto)
        and np.isclose(modelo.coef_[0], pendiente)
    ):
        raise RuntimeError("La implementación OLS no coincide con las ecuaciones normales.")

    pred_train = modelo.predict(x_train.reshape(-1, 1))
    residuos_train = y_train - pred_train
    suma_residuos_cuadrados = float(np.sum(residuos_train ** 2))
    r2_train = float(r2_score(y_train, pred_train))
    error_tipico = float(np.sqrt(suma_residuos_cuadrados / (len(train) - 2)))

    x_test = test["critic_score"].to_numpy(dtype=float).reshape(-1, 1)
    y_test = test["global_sales"].to_numpy(dtype=float)
    pred_test = modelo.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    metricas_test = {
        "filas": int(len(test)),
        "anio_minimo": int(test["year"].min()),
        "anio_maximo": int(test["year"].max()),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, pred_test)),
        "predicciones_ventas_negativas": int((pred_test < 0).sum()),
    }

    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)
    salida = test[["year", "critic_score", "global_sales"]].copy()
    salida = salida.rename(columns={"global_sales": "ventas_reales"})
    salida["ventas_estimadas"] = pred_test
    salida["residuo"] = salida["ventas_reales"] - salida["ventas_estimadas"]
    salida.to_csv(RUTA_PREDICCIONES, index=False)

    resultados = {
        "objetivo": "global_sales en millones",
        "predictor": "critic_score (nota disponible después del lanzamiento)",
        "ecuacion": f"ventas_estimadas = {intercepto:.8f} + {pendiente:.8f} * critic_score",
        "coeficientes": {
            "intercepto_a0": intercepto,
            "pendiente_a1": pendiente,
        },
        "entrenamiento": {
            "division": f"year < {ANIO_INICIO_TEST}",
            "filas": int(len(train)),
            "anio_minimo": int(train["year"].min()),
            "anio_maximo": int(train["year"].max()),
            "suma_residuos_cuadrados": suma_residuos_cuadrados,
            "r2": r2_train,
            "error_tipico_estimacion": error_tipico,
        },
        "prueba_temporal": metricas_test,
        "advertencia": (
            "Análisis histórico: critic_score no está disponible antes del "
            "lanzamiento y esta regresión no es el predictor pre-lanzamiento."
        ),
    }
    RUTA_METRICAS.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figura, eje = plt.subplots(figsize=(10, 6))
    eje.scatter(
        train["critic_score"], train["global_sales"],
        alpha=0.25, s=18, label="Entrenamiento (< 2012)",
    )
    eje.scatter(
        test["critic_score"], test["global_sales"],
        alpha=0.3, s=18, label="Prueba (2012+)",
    )
    linea_x = np.linspace(float(df["critic_score"].min()),
                          float(df["critic_score"].max()), 200)
    eje.plot(
        linea_x,
        modelo.predict(linea_x.reshape(-1, 1)),
        color="crimson",
        linewidth=2,
        label="Recta OLS ajustada con train",
    )
    eje.set(
        title="Nota de crítica y ventas globales: ajuste lineal histórico",
        xlabel="Nota de crítica",
        ylabel="Ventas globales (millones)",
    )
    eje.legend()
    eje.grid(alpha=0.2)
    figura.tight_layout()
    figura.savefig(RUTA_GRAFICO, dpi=150)
    plt.close(figura)

    print(f"Filas con crítica: {len(df):,} | train: {len(train):,} | test: {len(test):,}")
    print(f"Modelo: ventas = {intercepto:.4f} + {pendiente:.4f} * critic_score")
    print(f"R² train: {r2_train:.3f} | error típico: {error_tipico:.3f} millones")
    print(
        f"Test {ANIO_INICIO_TEST}+: MAE {metricas_test['mae']:.3f}"
        f" | RMSE {rmse:.3f} | R² {metricas_test['r2']:.3f}"
    )
    print("Artefactos:")
    for ruta in (RUTA_MODELO, RUTA_METRICAS, RUTA_PREDICCIONES, RUTA_GRAFICO):
        print(f"  {ruta.relative_to(RAIZ)}")
    print(
        "Nota: critic_score se conoce después del lanzamiento; "
        "este análisis no se usa para la predicción pre-lanzamiento."
    )


if __name__ == "__main__":
    main()
