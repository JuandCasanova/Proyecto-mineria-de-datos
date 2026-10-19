# Diseño de la matriz de características y exclusión de fuga
# Proyecto: predicción del éxito en un videojuego por salir
# Ejecutar desde la raíz del proyecto:
#   python src/04_transformacion/transformacion.py
#
# Entrada (datos_procesados/):
#   videojuegos_integrado.csv   16.598 x 23   las dos fuentes en una
# Salida: consola (diseño y auditoría de fuga; no escribe archivos).
# La implementación que aplica este diseño vive en:
#   src/04_transformacion/escalamiento.py
#   -> train_escalado.csv / test_escalado.csv
#
# FASE CRISP-DM: 3 · Preparación de los datos (transformación)
# Etapa KDD:      2 · Preprocesamiento (diseño de la matriz)
#
# ORDEN: carga -> exclusión por fuga -> diseño de la matriz -> qué hace
#        escalamiento.py con cada columna -> TODOs
#
# Este script NO transforma: decide y justifica. escalamiento.py ejecuta
# la decisión (split -> imputación -> StandardScaler -> one-hot).

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ / "datos_procesados" / "videojuegos_integrado.csv"

# Columnas que NO pueden entrar en X: son la respuesta o derivados de ella.
#   global_sales y las ventas por región: son ventas acumuladas; el modelo
#     las usa para predecir... las mismas ventas.
#   rank: es la posición en un ranking de ventas, la respuesta ordenada.
#   name / name_clave: identificadores, no rasgos; memorizarlos es fuga.
#   exito: es y.
COLUMNAS_FUGA = ["global_sales", "na_sales", "eu_sales", "jp_sales",
                 "other_sales", "rank", "name", "name_clave", "exito",
                 "genre_clave"]

# Categóricas de baja cardinalidad: entran con one-hot.
CATEGORICAS = ["platform", "genre", "rating"]

# Categóricas de alta cardinalidad: FUERA de la matriz. One-hot de 1.696
# developers sobre 16.598 filas serían columnas casi todas en cero; si el
# modelo lo necesita, se resuelven con target/frequency encoding.
CATEGORICAS_ALTA_CARDINALIDAD = ["publisher", "developer"]

# Numéricas con StandardScaler, ajustado SOLO con train (ver escalamiento.py).
# year entra al escaler como el resto: en CATALOGO_FUENTES.md §7 ya se midió
# que el RobustScaler empeora las variables con cola larga.
NUMERICAS_ESCALAR = ["year", "critic_score", "critic_count", "user_score",
                     "user_count"]

BANDERAS_NULOS = ["sin_critica", "sin_usuarios", "sin_desarrollador"]


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} . {texto}")


def main():
    print("=" * 70)
    print(" FASE 4 CRISP-DM . DISEÑO DE LA MATRIZ DE CARACTERÍSTICAS")
    print("=" * 70)

    titulo(0, "CARGA DEL DATASET INTEGRADO")
    if not RUTA_DATASET.exists():
        raise SystemExit(f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
                         "Ejecuta primero: python src/02_preparacion/merged.py")
    df = pd.read_csv(RUTA_DATASET)
    print(f" dataset: {RUTA_DATASET.relative_to(RAIZ)}")
    print(f" dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    titulo(1, "EXCLUSIÓN DE COLUMNAS CON FUGA DE DATOS")
    presentes = [c for c in COLUMNAS_FUGA if c in df.columns]
    ausentes = [c for c in COLUMNAS_FUGA if c not in df.columns]
    print(f" excluidas de X: {presentes}")
    if ausentes:
        print(f" [AVISO] no estaban en el dataset: {ausentes}")
    print(f" excluidas por alta cardinalidad: {CATEGORICAS_ALTA_CARDINALIDAD}")

    y = df["exito"]
    X = df.drop(columns=presentes)
    X = X.drop(columns=[c for c in CATEGORICAS_ALTA_CARDINALIDAD
                        if c in X.columns])
    print(f" X: {X.shape[0]} filas x {X.shape[1]} columnas")
    print(f" y: {y.name} (0/1), balance {y.value_counts().to_dict()}")

    titulo(2, "DISEÑO DE LA MATRIZ DE CARACTERÍSTICAS")
    print(" one-hot (baja cardinalidad, handle_unknown='ignore'):")
    for col in CATEGORICAS:
        if col in X.columns:
            print(f"   {col:<18} {X[col].nunique():>4} niveles")
    print(" banderas binarias (0/1, sin escalar):")
    for col in BANDERAS_NULOS:
        if col in X.columns:
            print(f"   {col:<18} {sorted(X[col].dropna().unique().tolist())}")
    print(" numéricas con StandardScaler (ajustado solo con train):")
    for col in NUMERICAS_ESCALAR:
        if col in X.columns:
            nulos = int(X[col].isna().sum())
            extra = " + bandera falta_*" if nulos else ""
            print(f"   {col:<18} nulos {nulos:>5}{extra} "
                  f"(imputación mediana, sin inventar evidencia)")

    titulo(3, "QUÉ HACE escalamiento.py CON ESTO")
    print("""
 split estratificado (semilla 42, test 20 %) ANTES de cualquier ajuste
   -> SimpleImputer(mediana) sobre las 5 numéricas, SOLO con train
   -> StandardScaler sobre las mismas 5, SOLO con train
   -> OneHotEncoder(handle_unknown='ignore') sobre platform/genre/rating
   -> sin_* y falta_* pasan como binarias sin escalar
   -> escribe datos_procesados/train_escalado.csv y test_escalado.csv
""")

    titulo(4, "PENDIENTES (por hacer, con el porqué)")
    print("""
 [ ] Target/frequency encoding de publisher/developer SOLO si el modelo
     lo necesita y gana en validación; ahora mismo están fuera por
     cardinalidad (566 y 1.696 niveles)
 [ ] Revisar con el modelo entrenado si las 53 columnas de one-hot de
     plataforma convienen o si conviene agrupar consolas por generación
""")

    print("=" * 70)
    print(" SIGUIENTE PASO: python src/04_transformacion/escalamiento.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
