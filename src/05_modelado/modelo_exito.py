"""Entrena y evalúa probabilidades de éxito con variables disponibles antes
del lanzamiento de un videojuego.

Ejecutar desde la raíz:
    python src/05_modelado/modelo_exito.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_PROCESADOS = RAIZ / "datos_procesados"
RUTA_DATASET = CARPETA_PROCESADOS / "videojuegos_integrado.csv"
RUTA_MODELO = CARPETA_PROCESADOS / "modelo_exito_pre_lanzamiento.joblib"
RUTA_METRICAS = CARPETA_PROCESADOS / "metricas_pre_lanzamiento.json"
RUTA_PREDICCIONES = CARPETA_PROCESADOS / "predicciones_test_pre_lanzamiento.csv"

SEMILLA = 42
ANIO_INICIO_TEST = 2012
NUM_SPLITS_CV = 5
UMBRAL_CLASIFICACION = 0.5
UMBRAL_EXITO_VENTAS = 1.0
NUMERICAS = ["year"]
CATEGORICAS = ["platform", "genre", "rating", "publisher", "developer"]
PREDICTORAS = NUMERICAS + CATEGORICAS


def construir_preprocesador():
    """Imputa y codifica dentro del pipeline para evitar fuga entre folds."""
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
        )),
    ])
    return ColumnTransformer([
        ("numerico", numerico, NUMERICAS),
        ("categorico", categorico, CATEGORICAS),
    ])


def construir_modelos():
    """Candidatos iniciales, ambos compensan el desbalance de clases."""
    return {
        "regresion_logistica": Pipeline([
            ("preprocesar", construir_preprocesador()),
            ("clasificar", LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                random_state=SEMILLA,
                solver="liblinear",
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocesar", construir_preprocesador()),
            ("clasificar", RandomForestClassifier(
                class_weight="balanced_subsample",
                max_features="sqrt",
                min_samples_leaf=3,
                n_estimators=300,
                n_jobs=-1,
                random_state=SEMILLA,
            )),
        ]),
    }


def evaluar_probabilidades(y_real, probabilidades, predicciones):
    """Calcula métricas de ranking, calibración y clasificación al umbral fijado."""
    matriz = confusion_matrix(y_real, predicciones, labels=[0, 1])
    return {
        "exactitud": float(accuracy_score(y_real, predicciones)),
        "precision_exito": float(
            precision_score(y_real, predicciones, zero_division=0)
        ),
        "recall_exito": float(
            recall_score(y_real, predicciones, zero_division=0)
        ),
        "f1_exito": float(f1_score(y_real, predicciones, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_real, probabilidades)),
        "average_precision": float(
            average_precision_score(y_real, probabilidades)
        ),
        "brier_score": float(brier_score_loss(y_real, probabilidades)),
        "matriz_confusion_vn_fp_fn_vp": matriz.tolist(),
    }


def main():
    print("=" * 72)
    print("MODELO SUPERVISADO · PROBABILIDAD DE ÉXITO ANTES DEL LANZAMIENTO")
    print("=" * 72)
    if not RUTA_DATASET.exists():
        raise SystemExit(
            f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
            "Ejecuta primero: python src/02_preparacion/merged.py"
        )

    df = pd.read_csv(RUTA_DATASET)
    requeridas = PREDICTORAS + ["exito"]
    faltantes = [columna for columna in requeridas if columna not in df.columns]
    if faltantes:
        raise SystemExit(f"[ERROR] Faltan columnas requeridas: {faltantes}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["exito"] = pd.to_numeric(df["exito"], errors="coerce")
    if df[["year", "exito"]].isna().any().any():
        raise SystemExit("[ERROR] year y exito no pueden contener valores vacíos.")
    if not set(df["exito"].unique()).issubset({0, 1}):
        raise SystemExit("[ERROR] exito debe contener únicamente 0 y 1.")

    X = df[PREDICTORAS].copy()
    for columna in CATEGORICAS:
        X[columna] = X[columna].astype(object).where(X[columna].notna(), np.nan)
    y = df["exito"].astype(int)

    # Mantiene años completos juntos: el conjunto de prueba representa juegos
    # posteriores a los usados para entrenar.
    mascara_test = df["year"] >= ANIO_INICIO_TEST
    X_train, X_test = X.loc[~mascara_test], X.loc[mascara_test]
    y_train, y_test = y.loc[~mascara_test], y.loc[mascara_test]
    if X_train.empty or X_test.empty or y_train.nunique() < 2:
        raise SystemExit(
            "[ERROR] La división temporal no produjo train/test con ambas "
            f"clases; revisa ANIO_INICIO_TEST={ANIO_INICIO_TEST}."
        )
    if y_test.nunique() < 2:
        raise SystemExit(
            "[ERROR] El periodo de prueba debe contener éxitos y no éxitos "
            "para calcular las métricas."
        )

    print(f"Dataset: {len(df):,} juegos | éxitos: {int(y.sum()):,} ({y.mean():.1%})")
    print(f"Variables previas al lanzamiento: {PREDICTORAS}")
    excluidas = [columna for columna in df.columns
                 if columna not in PREDICTORAS + ["exito"]]
    print(f"Excluidas del modelo (incluye ventas y reseñas posteriores): {excluidas}")
    print(
        f"Train: {len(X_train):,} juegos (hasta {int(df.loc[~mascara_test, 'year'].max())})"
        f" | éxitos {int(y_train.sum()):,} ({y_train.mean():.1%})"
    )
    print(
        f"Test : {len(X_test):,} juegos ({ANIO_INICIO_TEST}+)"
        f" | éxitos {int(y_test.sum()):,} ({y_test.mean():.1%})"
    )

    cv = StratifiedKFold(
        n_splits=NUM_SPLITS_CV,
        shuffle=True,
        random_state=SEMILLA,
    )
    candidatos = construir_modelos()
    resultados_cv = {}
    print(f"\nValidación cruzada estratificada ({NUM_SPLITS_CV} folds, solo train):")
    for nombre, modelo in candidatos.items():
        scores = cross_validate(
            modelo,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
            error_score="raise",
        )
        resultados_cv[nombre] = {
            "average_precision_media": float(
                scores["test_average_precision"].mean()
            ),
            "average_precision_desviacion": float(
                scores["test_average_precision"].std(ddof=1)
            ),
            "roc_auc_media": float(scores["test_roc_auc"].mean()),
            "roc_auc_desviacion": float(scores["test_roc_auc"].std(ddof=1)),
        }
        print(
            f"  {nombre:<24} AP "
            f"{resultados_cv[nombre]['average_precision_media']:.3f} "
            f"(±{resultados_cv[nombre]['average_precision_desviacion']:.3f})"
            f" | ROC-AUC {resultados_cv[nombre]['roc_auc_media']:.3f}"
        )

    mejor_nombre = max(
        resultados_cv,
        key=lambda nombre: resultados_cv[nombre]["average_precision_media"],
    )
    print(f"\nModelo seleccionado por Average Precision: {mejor_nombre}")

    # El calibrador se ajusta solo con train; sus probabilidades se validan
    # después en años posteriores que no participaron en ningún ajuste.
    modelo_calibrado = CalibratedClassifierCV(
        estimator=candidatos[mejor_nombre],
        method="sigmoid",
        cv=cv,
    )
    modelo_calibrado.fit(X_train, y_train)
    probabilidades = modelo_calibrado.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= UMBRAL_CLASIFICACION).astype(int)

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    prob_baseline = baseline.predict_proba(X_test)[:, 1]
    pred_baseline = baseline.predict(X_test)
    metricas_baseline = evaluar_probabilidades(
        y_test, prob_baseline, pred_baseline
    )
    metricas_modelo = evaluar_probabilidades(
        y_test, probabilidades, predicciones
    )
    print("\nEvaluación final en el periodo de prueba:")
    print(
        f"  Dummy prior       AP {metricas_baseline['average_precision']:.3f}"
        f" | Brier {metricas_baseline['brier_score']:.3f}"
    )
    print(
        f"  {mejor_nombre:<18} AP {metricas_modelo['average_precision']:.3f}"
        f" | ROC-AUC {metricas_modelo['roc_auc']:.3f}"
        f" | Brier {metricas_modelo['brier_score']:.3f}"
    )
    print(
        f"  Umbral 0,50: precisión {metricas_modelo['precision_exito']:.1%}"
        f" | recall {metricas_modelo['recall_exito']:.1%}"
        f" | F1 {metricas_modelo['f1_exito']:.3f}"
    )
    print(
        "  Matriz [[VN, FP], [FN, VP]]:",
        metricas_modelo["matriz_confusion_vn_fp_fn_vp"],
    )

    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_calibrado, RUTA_MODELO, compress=3)
    columnas_salida = [
        columna for columna in ["name"] + PREDICTORAS if columna in df.columns
    ]
    salida_test = df.loc[mascara_test, columnas_salida].copy()
    salida_test["exito_real"] = y_test
    salida_test["probabilidad_exito"] = probabilidades
    salida_test["prediccion_umbral_0_5"] = predicciones
    salida_test.to_csv(RUTA_PREDICCIONES, index=False)

    metricas = {
        "objetivo": (
            f"exito=1 si global_sales >= {UMBRAL_EXITO_VENTAS} millones"
        ),
        "momento_prediccion": "antes del lanzamiento",
        "predictores": PREDICTORAS,
        "columnas_excluidas": excluidas,
        "division": {
            "tipo": "temporal",
            "anio_inicio_test": ANIO_INICIO_TEST,
            "train_filas": int(len(X_train)),
            "train_anio_maximo": int(df.loc[~mascara_test, "year"].max()),
            "train_tasa_exito": float(y_train.mean()),
            "test_filas": int(len(X_test)),
            "test_tasa_exito": float(y_test.mean()),
        },
        "validacion_cruzada": {
            "tipo": "StratifiedKFold",
            "folds": NUM_SPLITS_CV,
            "criterio_seleccion": "average_precision",
            "modelos": resultados_cv,
        },
        "modelo_seleccionado": mejor_nombre,
        "calibracion_probabilidades": "sigmoid, ajustada solo en train",
        "umbral_clasificacion": UMBRAL_CLASIFICACION,
        "metricas_test": {
            "baseline_dummy_prior": metricas_baseline,
            "modelo": metricas_modelo,
        },
    }
    RUTA_METRICAS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nArtefactos:")
    for ruta in (RUTA_MODELO, RUTA_METRICAS, RUTA_PREDICCIONES):
        print(f"  {ruta.relative_to(RAIZ)}")
    print(
        "\nLa probabilidad requiere variables disponibles antes del lanzamiento. "
        "La calidad en años futuros depende de que los datos históricos sean representativos."
    )


if __name__ == "__main__":
    main()
