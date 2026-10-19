# Modelo de clasificación del éxito y evaluación contra la línea base
# Proyecto: predicción del éxito en un videojuego por salir
# Ejecutar desde la raíz del proyecto:
#   python src/05_modelado/modelo_exito.py
#
# Entrada (datos_procesados/):
#   videojuegos_integrado.csv   16.598 x 23   las dos fuentes en una
# Salidas pendientes (datos_procesados/):
#   modelo_exito.joblib          modelo entrenado
#   metrics_exito.json           métricas por modelo
#
# FASE CRISP-DM: 4 · Modelado  |  5 · Evaluación
# Etapa KDD:      4 · Modelado
#
# ORDEN: carga -> fuga -> división estratificada -> línea base
#        -> métricas -> TODOs
#
# Estado: PARCIAL. La línea base ya se calcula de verdad (DummyClassifier
# sobre el mismo split que usará el modelo real); el modelo real está
# pendiente. Sin línea base, cualquier exactitud del 87 % parece buena.

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ / "datos_procesados" / "videojuegos_integrado.csv"

COLUMNAS_FUGA = ["global_sales", "na_sales", "eu_sales", "jp_sales",
                 "other_sales", "rank", "name", "name_clave", "exito"]

SEMILLA = 42
TAMANO_TEST = 0.20


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} . {texto}")


def metricas(nombre, y_real, y_pred):
    """Exactitud, precisión, recall y F1 de la clase 1 (la que importa)."""
    cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
    print(f"\n {nombre}")
    print(f"   exactitud           {accuracy_score(y_real, y_pred):>8.1%}")
    print(f"   precisión (clase 1) {precision_score(y_real, y_pred, zero_division=0):>8.1%}")
    print(f"   recall    (clase 1) {recall_score(y_real, y_pred, zero_division=0):>8.1%}")
    print(f"   F1        (clase 1) {f1_score(y_real, y_pred, zero_division=0):>8.1%}")
    print(f"   matriz de confusión [[VN FP] [FN VP]] = {cm.tolist()}")
    print("   (el conjunto es 87,5 % clase 0: la exactitud sola engaña)")


def main():
    print("=" * 70)
    print(" FASE 5 CRISP-DM . MODELADO PREDICTIVO Y EVALUACIÓN")
    print(" Objetivo: superar la línea base de 87,5 % SIN perder recall")
    print("=" * 70)

    titulo(0, "CARGA Y SEPARACIÓN X / y")
    if not RUTA_DATASET.exists():
        raise SystemExit(f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
                         "Ejecuta primero: python src/02_preparacion/merged.py")
    df = pd.read_csv(RUTA_DATASET)
    print(f" dataset: {RUTA_DATASET.relative_to(RAIZ)}  ({df.shape})")

    fuga = [c for c in COLUMNAS_FUGA if c in df.columns]
    y = df["exito"]
    X = df.drop(columns=fuga)
    print(f" excluidas por fuga: {fuga}")
    print(f" X: {X.shape[1]} columnas | y: {int(y.sum())} éxitos "
          f"({y.mean():.1%})")

    titulo(1, "DIVISIÓN ESTRATIFICADA TRAIN / TEST")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_TEST, random_state=SEMILLA, stratify=y)
    print(f" semilla: {SEMILLA} | test: {TAMANO_TEST:.0%}")
    print(f" train: {len(X_train)} filas | éxitos {int(y_train.sum())} "
          f"({y_train.mean():.1%})")
    print(f" test : {len(X_test)} filas | éxitos {int(y_test.sum())} "
          f"({y_test.mean():.1%})")
    print(" la estratificación mantiene la misma proporción de éxitos en "
          "las dos mitades")

    titulo(2, "LÍNEA BASE (DummyClassifier: siempre predice la clase mayoritaria)")
    baseline = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    metricas("Línea base 'siempre no éxito'", y_test, y_base)

    titulo(3, "MODELO REAL (pendiente)")
    print("""
 [ ] Preprocesador ColumnTransformer en el pipeline:
       - OneHotEncoder(handle_unknown="ignore") sobre platform, genre, rating
       - StandardScaler sobre critic_score, critic_count, user_count
       - imputación de nulos de recepción con add_indicator=True (no se
         rellenan con la media: se conserva el patrón MNAR y se añade flag)
 [ ] RandomForestClassifier(class_weight="balanced", random_state=42)
     como primer modelo; comparar con Regresión Logística (baseline lineal)
     y con HistGradientBoosting si el RandomForest se queda corto
 [ ] Validación cruzada estratificada (k=5) sobre train; el test solo se
     toca UNA vez, al final
 [ ] Métricas obligatorias: recall y F1 de la clase 1, matriz de confusión,
     ROC-AUC y PR-AUC; la exactitud se reporta pero NO decide
 [ ] Importancia de variables para responder la pregunta de negocio:
     ¿qué hace que un juego supere el millón de unidades?
 [ ] Persistir el mejor modelo con joblib en datos_procesados/
""")

    print("=" * 70)
    print(" El pipeline completo se ejecuta con:  python main.py --all")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
