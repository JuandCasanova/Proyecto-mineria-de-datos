# Escalamiento y codificación de características para modelado
# Proyecto: predicción del éxito en un videojuego por salir
# Ejecutar desde la raíz del proyecto:
#   python src/04_transformacion/escalamiento.py
#
# Entrada (datos_procesados/):
#   videojuegos_integrado.csv   16.598 x 23   las dos fuentes en una
# Salidas (datos_procesados/):
#   train_escalado.csv           13.278 filas   split de entrenamiento transformado
#   test_escalado.csv             3.320 filas   split de prueba transformado
#
# FASE CRISP-DM: 3 · Preparación de los datos (transformación)
# Etapa KDD:      2 · Preprocesamiento (imputación, escalamiento, one-hot)
#
# ORDEN: carga -> selección blanca de columnas -> split ESTRATIFICADO
#        -> ajuste SOLO en train -> transformar -> verificar -> escribir
#
# REGLA DE ORO de este script: el ColumnTransformer se ajusta únicamente
# con el split de entrenamiento y luego se aplica a train y a test. Ajustar
# con los dos splits (o con todo el dataset) es fuga de información: las
# medias y desviaciones del scaler se calculan con datos que el modelo
# debería estar evaluando a ciegas.
#
# Decisiones que ya están tomadas en CATALOGO_FUENTES.md §7 y se aplican aquí:
#   - StandardScaler, no RobustScaler: user_count tiene asimetría 8,87 y
#     dividir por un IQR pequeño amplifica la cola en vez de controlarla.
#   - Los nulos de recepción NO se imputan a mano: SimpleImputer pone la
#     mediana solo para que el scaler pueda operar, y las banderas binarias
#     'falta_<col>' construidas ANTES de imputar conservan el patrón MNAR.
#     Se construyen a mano y no con add_indicator=True porque el scaler
#     posterior estandarizaría las banderas y dejarían de leerse como
#     proporciones (y sin_* ya existía con la misma lógica).
#   - publisher/developer fuera: 566 y 1.696 niveles de one-hot sobre
#     16.598 filas serían columnas densas de un solo cero.

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ / "datos_procesados" / "videojuegos_integrado.csv"
RUTA_TRAIN = RAIZ / "datos_procesados" / "train_escalado.csv"
RUTA_TEST = RAIZ / "datos_procesados" / "test_escalado.csv"

SEMILLA = 42
TAMANO_TEST = 0.20

# Selección blanca: SOLO estas columnas entran a X. Cualquier columna nueva
# que aparezca en el dataset queda fuera por defecto, que es lo opuesto a
# dropear por nombre y es la garantía contra fugas futuras.
COLUMNAS_FUGA = ["global_sales", "na_sales", "eu_sales", "jp_sales",
                 "other_sales", "rank", "name", "name_clave", "exito",
                 "genre_clave", "publisher", "developer"]

NUMERICAS = ["year", "critic_score", "critic_count", "user_score", "user_count"]
CATEGORICAS = ["platform", "genre", "rating"]
BINARIAS = ["sin_critica", "sin_usuarios", "sin_desarrollador"]
# year no tiene nulos en el integrado; estas 4 sí (recepción MNAR)
COLUMNAS_CON_NULOS = ["critic_score", "critic_count", "user_score", "user_count"]
INDICADORES = [f"falta_{c}" for c in COLUMNAS_CON_NULOS]


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} . {texto}")


def construir_preprocesador():
    """Imputación+escalamiento de numéricas y one-hot de categóricas."""
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numerico, NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAS),
        ],
        remainder="passthrough",   # banderas sin_* y falta_* pasan tal cual
        verbose_feature_names_out=False,
    )


def nombres_columnas(preprocesador):
    nombres = list(preprocesador.get_feature_names_out())
    # con verbose_feature_names_out=False no hay prefijos 'num_'/'cat_'
    return [n.replace("num_", "").replace("cat_", "").replace("remainder_", "")
            for n in nombres]


def verificar(train, test, columnas_num):
    """Media/desviación por columna escalada: train ~0/1, test parecido."""
    print(f"\n train: {train.shape[0]} filas x {train.shape[1]} columnas | "
          f"éxitos {int(train['exito'].sum())} ({train['exito'].mean():.1%})")
    print(f" test : {test.shape[0]} filas x {test.shape[1]} columnas | "
          f"éxitos {int(test['exito'].sum())} ({test['exito'].mean():.1%})")

    print("\n Verificación del escalamiento (ajustado SOLO con train):")
    print(f"   {'columna':<24} {'media train':>12} {'std train':>10} "
          f"{'media test':>11}")
    print("   " + "-" * 60)
    for col in columnas_num:
        if col in train.columns:
            print(f"   {col:<24} {train[col].mean():>12.3f} "
                  f"{train[col].std(ddof=0):>10.3f} {test[col].mean():>11.3f}")

    indicadores = INDICADORES + BINARIAS
    print("\n Banderas binarias SIN escalar (proporción de nulos / marcas):")
    print(f"   {'columna':<34} {'train':>7} {'test':>7}")
    print("   " + "-" * 50)
    for col in indicadores:
        if col in train.columns:
            print(f"   {col:<34} {train[col].mean():>6.1%} "
                  f"{test[col].mean():>6.1%}")

    print("\n Las medias de test están cerca de las de train porque test")
    print(" se transformó con el scaler de train: si divergieran mucho, el")
    print(" split no es representativo (no porque haya fuga).")


def main():
    print("=" * 70)
    print(" FASE 4 CRISP-DM . ESCALAMIENTO Y CODIFICACIÓN DE CARACTERÍSTICAS")
    print(" StandardScaler (train) + OneHotEncoder + split estratificado")
    print("=" * 70)

    titulo(0, "CARGA Y SELECCIÓN BLANCA DE COLUMNAS")
    if not RUTA_DATASET.exists():
        raise SystemExit(f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
                         "Ejecuta primero: python src/02_preparacion/merged.py")
    df = pd.read_csv(RUTA_DATASET)
    print(f" dataset: {RUTA_DATASET.relative_to(RAIZ)}  ({df.shape})")

    fuga_detectada = [c for c in COLUMNAS_FUGA if c in df.columns]
    X = df[NUMERICAS + CATEGORICAS + BINARIAS].copy()
    # Banderas MNAR construidas ANTES de imputar: registran dónde FALTABA
    # el dato, no lo que la mediana puso en su lugar.
    for col in COLUMNAS_CON_NULOS:
        X[f"falta_{col}"] = df[col].isna().astype(int)
    y = df["exito"]
    print(f" fuera de X (fuga o alto cardinalidad): {fuga_detectada}")
    print(f" numéricas   : {NUMERICAS}")
    print(f" categóricas : {CATEGORICAS}")
    print(f" binarias    : {BINARIAS} (sin escalar)")
    print(f" indicadores : {INDICADORES} (sin escalar, MNAR)")
    print(f" X: {X.shape[1]} columnas de entrada | y: exito "
          f"({y.mean():.1%} éxitos)")
    print(f" nulos en X  : {int(X.isna().sum().sum())} (recepción MNAR, "
          f"se imputan con mediana + indicador)")

    titulo(1, "DIVISIÓN ESTRATIFICADA ANTES DE ESCALAR")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_TEST, random_state=SEMILLA, stratify=y)
    print(f" semilla: {SEMILLA} | test: {TAMANO_TEST:.0%} | estratificada por exito")
    print(f" train {len(X_train)} filas ({y_train.mean():.1%} éxitos) | "
          f"test {len(X_test)} filas ({y_test.mean():.1%} éxitos)")
    print(" el split va ANTES del ajuste: el scaler nunca ve el test")

    titulo(2, "AJUSTE DEL PREPROCESADOR SOLO CON TRAIN")
    pre = construir_preprocesador()
    pre.fit(X_train)
    print(" ColumnTransformer ajustado con train = "
          f"{len(X_train)} filas")
    print("   num : SimpleImputer(mediana) -> StandardScaler")
    print("   cat : OneHotEncoder(handle_unknown='ignore')")
    print("   resto: banderas sin_* y falta_* pasan sin escalar")

    nombres = nombres_columnas(pre)
    train = pd.DataFrame(pre.transform(X_train), columns=nombres)
    test = pd.DataFrame(pre.transform(X_test), columns=nombres)
    train["exito"] = y_train.to_numpy()
    test["exito"] = y_test.to_numpy()

    titulo(3, "VERIFICACIÓN DE MEDIA Y DESVIACIÓN")
    verificar(train, test, NUMERICAS)

    titulo(4, "ESCRITURA DE LOS ARTIFACTOS")
    train.to_csv(RUTA_TRAIN, index=False)
    test.to_csv(RUTA_TEST, index=False)
    for ruta in (RUTA_TRAIN, RUTA_TEST):
        print(f"   {ruta.relative_to(RAIZ)}  "
              f"({ruta.stat().st_size / 1024:.1f} KB)")

    print("""
 Decisiones aplicadas (detalle en docs/CATALOGO_FUENTES.md):
   - StandardScaler sobre las 5 numéricas; ajustado solo con train.
   - Nulos de recepción: mediana + indicador, nunca valor inventado.
   - One-hot de platform/genre/rating con handle_unknown="ignore" para
     cuando lleguen plataformas nuevas en el videojuego por salir.
   - publisher/developer quedaron fuera: 566 y 1.696 niveles.
   - Las columnas de venta y rank no entraron: son la respuesta.
""")

    print("=" * 70)
    print(" SIGUIENTE PASO: python src/05_modelado/modelo_exito.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
