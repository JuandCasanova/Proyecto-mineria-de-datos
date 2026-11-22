"""Interpolación local de Lagrange sobre ventas promedio por año."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_PROCESADOS = RAIZ / "datos_procesados"
RUTA_DATASET = CARPETA_PROCESADOS / "videojuegos_integrado.csv"
RUTA_RESULTADO = CARPETA_PROCESADOS / "resultado_interpolacion_lagrange.csv"
RUTA_METRICAS = CARPETA_PROCESADOS / "metricas_interpolacion_lagrange.json"
RUTA_GRAFICO = CARPETA_PROCESADOS / "interpolacion_lagrange_ventas_anuales.png"

ANIO_OBJETIVO = 2013
MIN_JUEGOS_POR_ANIO = 100
CANTIDAD_NODOS = 3


def lagrange(xs, ys, x):
    """Evalúa P(x)=sum(y_i * L_i(x)); rechaza nodos repetidos."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs e ys deben tener la misma longitud y al menos 2 nodos.")
    if len(set(xs)) != len(xs):
        raise ValueError("Los nodos x deben ser distintos.")

    valor = 0.0
    pesos = []
    for i, (x_i, y_i) in enumerate(zip(xs, ys)):
        base = 1.0
        for j, x_j in enumerate(xs):
            if i != j:
                base *= (x - x_j) / (x_i - x_j)
        pesos.append(base)
        valor += y_i * base
    return float(valor), pesos


def construir_serie_anual(df):
    """Agrega ventas por año y descarta años con muy pocos juegos."""
    datos = df[["year", "global_sales"]].copy()
    datos["year"] = pd.to_numeric(datos["year"], errors="coerce")
    datos["global_sales"] = pd.to_numeric(
        datos["global_sales"], errors="coerce"
    )
    serie = (
        datos.dropna()
        .groupby("year")["global_sales"]
        .agg(ventas_promedio="mean", juegos="size")
        .sort_index()
    )
    return serie.loc[serie["juegos"] >= MIN_JUEGOS_POR_ANIO]


def main():
    print("=" * 72)
    print("MÉTODOS NUMÉRICOS · INTERPOLACIÓN POLINOMIAL DE LAGRANGE")
    print("=" * 72)
    if not RUTA_DATASET.exists():
        raise SystemExit(
            f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
            "Ejecuta primero: python src/02_preparacion/merged.py"
        )

    df = pd.read_csv(RUTA_DATASET, usecols=["year", "global_sales"])
    serie = construir_serie_anual(df)
    if ANIO_OBJETIVO not in serie.index:
        raise SystemExit(
            f"[ERROR] El año {ANIO_OBJETIVO} no tiene al menos "
            f"{MIN_JUEGOS_POR_ANIO} juegos para validar la interpolación."
        )

    # Se retira el año objetivo de los nodos: su valor real solo se usa
    # después para medir el error de esta demostración.
    disponibles = [int(anio) for anio in serie.index if anio != ANIO_OBJETIVO]
    nodos = sorted(
        disponibles,
        key=lambda anio: (abs(anio - ANIO_OBJETIVO), anio),
    )[:CANTIDAD_NODOS]
    nodos.sort()
    if len(nodos) != CANTIDAD_NODOS:
        raise SystemExit("[ERROR] No hay suficientes años observados distintos.")
    if not nodos[0] < ANIO_OBJETIVO < nodos[-1]:
        raise SystemExit(
            "[ERROR] El año objetivo debe estar dentro de los nodos; "
            "Lagrange no se usará para extrapolar."
        )

    xs = np.asarray(nodos, dtype=float)
    ys = serie.loc[nodos, "ventas_promedio"].to_numpy(dtype=float)
    estimacion, pesos = lagrange(xs, ys, float(ANIO_OBJETIVO))
    real = float(serie.loc[ANIO_OBJETIVO, "ventas_promedio"])
    error_absoluto = abs(real - estimacion)
    error_relativo = error_absoluto / abs(real) if real else None

    filas = []
    for anio, valor, peso in zip(nodos, ys, pesos):
        filas.append({
            "tipo": "nodo",
            "year": anio,
            "juegos": int(serie.loc[anio, "juegos"]),
            "ventas_promedio_real": float(valor),
            "peso_lagrange": float(peso),
            "ventas_promedio_estimada": float(valor),
        })
    filas.append({
        "tipo": "objetivo_validacion",
        "year": ANIO_OBJETIVO,
        "juegos": int(serie.loc[ANIO_OBJETIVO, "juegos"]),
        "ventas_promedio_real": real,
        "peso_lagrange": np.nan,
        "ventas_promedio_estimada": estimacion,
    })
    resultado = pd.DataFrame(filas)

    informe = {
        "metodo": "Interpolación polinomial de Lagrange",
        "variable_interpolada": "promedio de global_sales por año (millones)",
        "anio_objetivo": ANIO_OBJETIVO,
        "nodos": [
            {
                "year": int(anio),
                "juegos": int(serie.loc[anio, "juegos"]),
                "ventas_promedio": float(valor),
                "peso_en_objetivo": float(peso),
            }
            for anio, valor, peso in zip(nodos, ys, pesos)
        ],
        "grado_maximo": CANTIDAD_NODOS - 1,
        "estimacion": estimacion,
        "valor_real_reservado_para_validacion": real,
        "error_absoluto": float(error_absoluto),
        "error_relativo": float(error_relativo) if error_relativo is not None else None,
        "advertencia": (
            "Ejercicio sobre un promedio anual, no una predicción de ventas "
            "ni éxito de un videojuego individual. Se interpola dentro del "
            "rango de nodos y el año objetivo se excluyó de ellos."
        ),
    }

    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RUTA_RESULTADO, index=False)
    RUTA_METRICAS.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    grid = np.linspace(xs.min(), xs.max(), 200)
    curva = [lagrange(xs, ys, punto)[0] for punto in grid]
    figura, eje = plt.subplots(figsize=(10, 6))
    eje.plot(
        serie.index,
        serie["ventas_promedio"],
        color="steelblue",
        alpha=0.65,
        marker="o",
        markersize=4,
        label=f"Años con ≥{MIN_JUEGOS_POR_ANIO} juegos",
    )
    eje.plot(grid, curva, color="crimson", linewidth=2,
             label=f"Polinomio local (grado ≤ {CANTIDAD_NODOS - 1})")
    eje.scatter(xs, ys, color="darkorange", s=65, zorder=3, label="Nodos")
    eje.scatter(
        [ANIO_OBJETIVO], [real], color="black", marker="x", s=75, zorder=4,
        label="Valor real reservado",
    )
    eje.scatter(
        [ANIO_OBJETIVO], [estimacion], color="crimson", marker="D", s=45,
        zorder=4, label="Estimación Lagrange",
    )
    eje.set(
        title="Interpolación local del promedio de ventas por año",
        xlabel="Año",
        ylabel="Ventas globales promedio (millones)",
    )
    eje.legend()
    eje.grid(alpha=0.2)
    figura.tight_layout()
    figura.savefig(RUTA_GRAFICO, dpi=150)
    plt.close(figura)

    print(f"Año objetivo: {ANIO_OBJETIVO} (excluido de los nodos)")
    print("Nodos usados:")
    print(resultado.loc[resultado["tipo"] == "nodo"].to_string(index=False))
    print(f"Estimación: {estimacion:.4f} millones")
    print(f"Promedio real reservado: {real:.4f} millones")
    print(f"Error absoluto: {error_absoluto:.4f} millones")
    print("Artefactos:")
    for ruta in (RUTA_RESULTADO, RUTA_METRICAS, RUTA_GRAFICO):
        print(f"  {ruta.relative_to(RAIZ)}")
    print(
        "Esta interpolación es una demostración sobre promedios anuales; "
        "no estima el éxito de un juego individual."
    )


if __name__ == "__main__":
    main()
