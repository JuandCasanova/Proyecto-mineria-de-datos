"""Compara K-means y el árbol jerárquico aglomerativo Ward en juegos incompletos.

Ambos métodos se ajustan sin usar ventas ni la etiqueta exito. Se comparan
en una misma muestra con silueta y Adjusted Rand Index.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ / "datos_procesados" / "videojuegos_integrado.csv"
CARPETA_SALIDA = RAIZ / "datos_procesados"
RUTA_ASIGNACIONES = CARPETA_SALIDA / "juegos_incompletos_clusters.csv"
RUTA_RESUMEN = CARPETA_SALIDA / "resumen_clusters.csv"
RUTA_SILUETA = CARPETA_SALIDA / "evaluacion_kmeans.csv"
RUTA_JERARQUICO = CARPETA_SALIDA / "muestra_jerarquica.csv"
RUTA_DENDROGRAMA = CARPETA_SALIDA / "dendrograma_juegos_incompletos.png"
RUTA_COMPARACION = CARPETA_SALIDA / "comparacion_modelos_no_supervisados.csv"

SEMILLA = 42
K_MAXIMO = 8
MUESTRA_SILUETA = 2_000
MUESTRA_JERARQUICA = 1_000

NUMERICAS = ["year", "critic_score", "critic_count", "user_score", "user_count"]
CATEGORICAS = ["platform", "genre", "rating"]
COLUMNAS_RECEPCION = ["critic_score", "critic_count", "user_score", "user_count"]


def seleccionar_cohorte(df):
    """Incluye juegos sin rating o con información incompleta de reseñas."""
    if "rating" not in df.columns:
        raise ValueError("El dataset integrado debe contener la columna 'rating'.")

    rating_sin_clasificar = (
        df["rating"].isna()
        | df["rating"].astype("string").str.strip().str.casefold().eq("sin clasificar")
    )
    recepcion_incompleta = pd.Series(False, index=df.index)
    for bandera in ("sin_critica", "sin_usuarios"):
        if bandera in df.columns:
            recepcion_incompleta |= pd.to_numeric(
                df[bandera], errors="coerce"
            ).fillna(0).gt(0)
    for columna in COLUMNAS_RECEPCION:
        if columna in df.columns:
            recepcion_incompleta |= df[columna].isna()

    mask = rating_sin_clasificar | recepcion_incompleta
    cohorte = df.loc[mask].copy()
    razones = []
    for indice in cohorte.index:
        motivos = []
        if rating_sin_clasificar.loc[indice]:
            motivos.append("rating_sin_clasificar")
        if recepcion_incompleta.loc[indice]:
            motivos.append("recepcion_incompleta")
        razones.append("+".join(motivos))
    cohorte["motivo_inclusion"] = razones
    return cohorte


def preparar_caracteristicas(cohorte):
    """Prepara rasgos disponibles; excluye ventas, objetivo e identificadores."""
    numericas = [
        columna for columna in NUMERICAS
        if columna in cohorte.columns
        and pd.to_numeric(cohorte[columna], errors="coerce").notna().any()
    ]
    categoricas = [
        columna for columna in CATEGORICAS
        if columna in cohorte.columns and cohorte[columna].notna().any()
    ]
    if not numericas and not categoricas:
        raise ValueError("La cohorte no tiene variables utilizables para agrupar.")

    X = cohorte[numericas + categoricas].copy()
    for columna in numericas:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")

    transformadores = []
    if numericas:
        transformadores.append((
            "numericas",
            Pipeline([
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler()),
            ]),
            numericas,
        ))
    if categoricas:
        transformadores.append((
            "categoricas",
            Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )),
            ]),
            categoricas,
        ))
    preprocesador = ColumnTransformer(
        transformadores, remainder="drop", sparse_threshold=0
    )
    X_transformado = preprocesador.fit_transform(X)
    return X_transformado, numericas, categoricas


def seleccionar_k(X):
    """Evalúa k entre 2 y 8 con silueta y semilla reproducible."""
    if len(X) < 3:
        raise ValueError("Se necesitan al menos 3 juegos para agrupar.")

    resultados = []
    mejor = None
    for k in range(2, min(K_MAXIMO, len(X) - 1) + 1):
        modelo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10)
        etiquetas = modelo.fit_predict(X)
        if len(np.unique(etiquetas)) < 2:
            continue
        silueta = silhouette_score(
            X,
            etiquetas,
            sample_size=min(MUESTRA_SILUETA, len(X)),
            random_state=SEMILLA,
        )
        resultados.append({
            "k": k,
            "silueta": float(silueta),
            "inercia": float(modelo.inertia_),
        })
        if mejor is None or silueta > mejor["silueta"]:
            mejor = {"k": k, "silueta": float(silueta), "modelo": modelo}

    if mejor is None:
        raise ValueError("No fue posible obtener una partición válida para K-means.")
    return mejor, resultados


def resumir_clusters(cohorte, etiquetas):
    """Resume perfiles para orientar revisión humana, no como reglas causales."""
    datos = cohorte.copy()
    datos["cluster_kmeans"] = etiquetas
    filas = []
    for cluster, grupo in datos.groupby("cluster_kmeans", sort=True):
        fila = {
            "cluster_kmeans": int(cluster),
            "juegos": int(len(grupo)),
        }
        for columna in NUMERICAS:
            if columna in grupo.columns:
                fila[f"promedio_{columna}"] = pd.to_numeric(
                    grupo[columna], errors="coerce"
                ).mean()
        for columna in ("platform", "genre", "rating"):
            if columna in grupo.columns:
                moda = grupo[columna].mode(dropna=True)
                fila[f"moda_{columna}"] = moda.iloc[0] if not moda.empty else ""
        if "exito" in grupo.columns:
            fila["tasa_exito_observada"] = pd.to_numeric(
                grupo["exito"], errors="coerce"
            ).mean()
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    print("=" * 72)
    print("AGRUPAMIENTO NO SUPERVISADO · JUEGOS SIN CLASIFICACIÓN O RESEÑAS")
    print("=" * 72)
    if not RUTA_DATASET.exists():
        raise SystemExit(
            f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
            "Ejecuta primero: python src/02_preparacion/merged.py"
        )

    df = pd.read_csv(RUTA_DATASET)
    cohorte = seleccionar_cohorte(df)
    if len(cohorte) < 3:
        raise SystemExit(
            f"[ERROR] La cohorte tiene {len(cohorte)} filas; se requieren al menos 3."
        )
    cohorte.insert(0, "indice_original", cohorte.index)
    X, numericas, categoricas = preparar_caracteristicas(cohorte)
    mejor, evaluaciones = seleccionar_k(X)

    cohorte["cluster_kmeans"] = mejor["modelo"].labels_
    resumen = resumir_clusters(cohorte, mejor["modelo"].labels_)
    evaluacion = pd.DataFrame(evaluaciones)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    cohorte.to_csv(RUTA_ASIGNACIONES, index=False)
    resumen.to_csv(RUTA_RESUMEN, index=False)
    evaluacion.to_csv(RUTA_SILUETA, index=False)

    # El dendrograma es una muestra reproducible: Ward requiere memoria
    # cuadrática y no es práctico ejecutarlo sobre las miles de filas completas.
    muestra = cohorte.sample(
        n=min(MUESTRA_JERARQUICA, len(cohorte)),
        random_state=SEMILLA,
    )
    posiciones = cohorte.index.get_indexer(muestra.index)
    X_muestra = X[posiciones]
    modelo_kmeans_muestra = KMeans(
        n_clusters=mejor["k"],
        random_state=SEMILLA,
        n_init=10,
    )
    etiquetas_kmeans_muestra = modelo_kmeans_muestra.fit_predict(X_muestra)
    arbol = linkage(X_muestra, method="ward", metric="euclidean")
    etiquetas_arbol = fcluster(arbol, t=mejor["k"], criterion="maxclust")
    silueta_kmeans_muestra = silhouette_score(
        X_muestra, etiquetas_kmeans_muestra
    )
    silueta_jerarquico = silhouette_score(X_muestra, etiquetas_arbol)
    acuerdo_ari = adjusted_rand_score(
        etiquetas_kmeans_muestra, etiquetas_arbol
    )
    comparacion = pd.DataFrame([
        {
            "modelo": "K-means",
            "algoritmo": "particional",
            "filas_ajuste": len(muestra),
            "k": len(np.unique(etiquetas_kmeans_muestra)),
            "silueta_muestra": silueta_kmeans_muestra,
            "ari_vs_kmeans": None,
        },
        {
            "modelo": "Árbol jerárquico (Ward)",
            "algoritmo": "aglomerativo Ward",
            "filas_ajuste": len(muestra),
            "k": len(np.unique(etiquetas_arbol)),
            "silueta_muestra": silueta_jerarquico,
            "ari_vs_kmeans": acuerdo_ari,
        },
    ])
    identificadores = [
        columna for columna in ("indice_original", "name", "platform", "genre", "rating")
        if columna in muestra.columns
    ]
    salida_arbol = muestra[identificadores].copy()
    salida_arbol["cluster_kmeans"] = etiquetas_kmeans_muestra
    salida_arbol["cluster_jerarquico"] = etiquetas_arbol
    salida_arbol.to_csv(RUTA_JERARQUICO, index=False)
    comparacion.to_csv(RUTA_COMPARACION, index=False)

    figura, eje = plt.subplots(figsize=(14, 7))
    dendrogram(
        arbol,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45,
        leaf_font_size=9,
        show_contracted=True,
        ax=eje,
    )
    eje.set_title(
        f"Dendrograma Ward · muestra de {len(muestra)} juegos · "
        f"corte orientativo k={mejor['k']}"
    )
    eje.set_xlabel("Grupos de hojas (vista truncada)")
    eje.set_ylabel("Distancia euclidiana")
    figura.tight_layout()
    figura.savefig(RUTA_DENDROGRAMA, dpi=150)
    plt.close(figura)

    print(f"Dataset integrado: {len(df):,} filas")
    print(f"Cohorte seleccionada: {len(cohorte):,} filas")
    print(f"Variables usadas: {numericas + categoricas}")
    print(f"K elegido: {mejor['k']} | silueta: {mejor['silueta']:.3f}")
    print(f"Muestra jerárquica: {len(muestra):,} filas (semilla {SEMILLA})")
    print("\nComparación de ambos modelos en la misma muestra:")
    print(comparacion.round(3).to_string(index=False))
    print("\nPerfil de clusters:")
    print(resumen.round(3).to_string(index=False))
    print("\nArtefactos generados:")
    for ruta in (
        RUTA_ASIGNACIONES, RUTA_RESUMEN, RUTA_SILUETA,
        RUTA_JERARQUICO, RUTA_DENDROGRAMA, RUTA_COMPARACION,
    ):
        print(f"  {ruta.relative_to(RAIZ)}")
    print(
        "\nLa tasa de éxito del resumen es retrospectiva y descriptiva; "
        "no fue una variable de entrada ni implica causalidad."
    )


if __name__ == "__main__":
    main()
