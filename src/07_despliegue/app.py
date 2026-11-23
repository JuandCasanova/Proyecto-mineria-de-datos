"""Aplicación web pública para explorar y usar el proyecto de minería de datos."""

import hmac
import json
import mimetypes
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "datos"
PROCESADOS = RAIZ / "datos_procesados"
IMAGENES = RAIZ / "img"
RUTA_DATASET = PROCESADOS / "videojuegos_integrado.csv"
RUTA_METRICAS_MODELO = PROCESADOS / "metricas_pre_lanzamiento.json"
RUTA_MODELO = PROCESADOS / "modelo_exito_pre_lanzamiento.joblib"

st.set_page_config(
    page_title="Éxito de videojuegos",
    page_icon="🎮",
    layout="wide",
)

def aplicar_estilos():
    st.markdown(
        """
        <style>
        .stApp { background: #f4f6fb; color: #172033; }
        [data-testid="stHeader"] { background: rgba(244, 246, 251, .92); }
        .block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #151a32 0%, #101728 100%);
            border-right: 1px solid #252d45;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stSidebar"] label { color: #e7ebf5; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #d0d8e8 !important; opacity: 1 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 10px; padding: .4rem .6rem; margin-bottom: .15rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #222b43;
        }
        div[data-testid="stMetric"] {
            background: #fff; border: 1px solid #e5e9f2; border-radius: 16px;
            padding: 1rem 1.15rem; box-shadow: 0 5px 18px rgba(30, 41, 59, .04);
        }
        div[data-testid="stMetricLabel"] { color: #64748b; }
        .app-hero {
            background: linear-gradient(120deg, #20264a 0%, #3b3477 58%, #5143a3 100%);
            border-radius: 24px; padding: 2.2rem 2.4rem; margin: .4rem 0 1.3rem;
            color: #fff; box-shadow: 0 16px 38px rgba(44, 43, 107, .17);
        }
        .app-hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.1; margin: .4rem 0 .7rem; }
        .app-hero p { color: #e4e7ff; font-size: 1.05rem; max-width: 760px; margin-bottom: 0; }
        .hero-tag, .page-kicker {
            color: #a9f0df; font-size: .75rem; font-weight: 750;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .page-kicker {
            color: #3e309f !important; opacity: 1 !important;
            font-size: .8rem; font-weight: 800; margin-bottom: .35rem;
        }
        .page-description { color: #64748b; font-size: 1.04rem; margin-top: -.25rem; margin-bottom: 1.4rem; }
        .feature-card {
            background: #fff; border: 1px solid #e5e9f2; border-radius: 16px;
            padding: 1.1rem 1.2rem; min-height: 135px; box-shadow: 0 5px 18px rgba(30, 41, 59, .04);
        }
        .feature-card h3 { margin: 0 0 .45rem; color: #20264a; font-size: 1.04rem; }
        .feature-card p { margin: 0; color: #64748b; font-size: .94rem; }
        div.stButton > button { border-radius: 11px; font-weight: 650; min-height: 2.8rem; }
        div[data-testid="stTabs"] button { font-weight: 650; }
        div[data-testid="stAlert"] { border-radius: 13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def encabezado_pagina(etiqueta, titulo, descripcion):
    st.markdown(
        f"""
        <div class="page-kicker">{etiqueta}</div>
        <h1>{titulo}</h1>
        <p class="page-description">{descripcion}</p>
        """,
        unsafe_allow_html=True,
    )


def navegar_a(pagina):
    st.session_state["navegacion"] = pagina


@st.cache_data(show_spinner=False)
def cargar_csv(ruta):
    return pd.read_csv(ruta, low_memory=False)


@st.cache_data(show_spinner=False)
def cargar_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def mostrar_inicio(metricas):
    st.markdown(
        """
        <section class="app-hero">
          <div class="hero-tag">Explorador de datos · modelo pre-lanzamiento</div>
          <h1>¿Un videojuego podría superar el millón de ventas?</h1>
          <p>Consulta los datos históricos, entiende los análisis y estima una
          probabilidad usando solo información disponible antes del lanzamiento.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "La estimación es una referencia basada en datos históricos; no es una "
        "garantía de ventas ni una recomendación de inversión."
    )

    dataset = cargar_csv(str(RUTA_DATASET))
    metricas_modelo = metricas["metricas_test"]["modelo"]
    exito_count = int(dataset["exito"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Juegos", f"{len(dataset):,}", help="Registros del catálogo integrado.")
    c2.metric(
        "Éxitos",
        f"{exito_count:,}",
        help=f"{dataset['exito'].mean():.1%} del catálogo superó el umbral.",
    )
    c3.metric(
        "Entradas",
        str(len(metricas["predictores"])),
        help="Variables del predictor disponibles antes del lanzamiento.",
    )

    st.subheader("¿Qué quieres hacer?")
    acciones = st.columns(2)
    acciones[0].button(
        "🎯  Estimar un videojuego",
        type="primary",
        width="stretch",
        on_click=navegar_a,
        args=("Predicción",),
    )
    acciones[1].button(
        "📚  Explorar los datos",
        width="stretch",
        on_click=navegar_a,
        args=("Datos y descargas",),
    )

    st.markdown(
        """
        <div class="feature-card">
          <h3>🧭 Recorrido de la aplicación</h3>
          <p><b>Predicción:</b> evalúa un juego o carga un CSV. &nbsp;
          <b>Datos:</b> filtra el catálogo y descarga archivos. &nbsp;
          <b>Segmentación y métodos:</b> consulta los análisis complementarios.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Cómo se evaluó el predictor", expanded=False):
        division = metricas["division"]
        baseline = metricas["metricas_test"]["baseline_dummy_prior"]
        st.caption(
            f"Entrenamiento hasta {division['train_anio_maximo']}; prueba "
            f"temporal desde {division['anio_inicio_test']}."
        )
        a, b, c = st.columns(3)
        a.metric(
            "Average Precision · prueba",
            f"{metricas_modelo['average_precision']:.3f}",
            help=f"Línea base: {baseline['average_precision']:.3f}",
        )
        b.metric("ROC-AUC", f"{metricas_modelo['roc_auc']:.3f}")
        c.metric(
            "Brier score",
            f"{metricas_modelo['brier_score']:.3f}",
            help=f"Línea base: {baseline['brier_score']:.3f}; menor es mejor.",
        )
        st.info(
            "Average Precision evalúa el ordenamiento de los casos; no es la "
            "probabilidad individual de un videojuego. El recall al umbral 0,50 "
            f"fue {metricas_modelo['recall_exito']:.1%}."
        )

    figura = IMAGENES / "fig1_barras_exito.png"
    if figura.exists():
        with st.expander("Ver distribución histórica del éxito"):
            st.image(str(figura), width="stretch")


def opciones_categoria(dataset, columna):
    valores = dataset[columna].dropna().astype(str).str.strip()
    return sorted(valor for valor in valores.unique() if valor)


def formulario_prediccion(metricas, dataset):
    columnas = metricas["predictores"]
    st.markdown("#### Describe el juego")
    st.caption("Completa lo que ya se conoce antes de su lanzamiento.")
    with st.form("formulario_prediccion"):
        c1, c2 = st.columns(2)
        year = c1.number_input(
            "Año de lanzamiento",
            min_value=1980,
            max_value=2035,
            value=2020,
            step=1,
        )
        platform = c2.selectbox(
            "Plataforma",
            opciones_categoria(dataset, "platform"),
            help="Selecciona la consola o plataforma prevista.",
        )
        c3, c4 = st.columns(2)
        genre = c3.selectbox("Género", opciones_categoria(dataset, "genre"))
        ratings = opciones_categoria(dataset, "rating")
        rating_default = (
            ratings.index("Sin clasificar")
            if "Sin clasificar" in ratings
            else 0
        )
        rating = c4.selectbox(
            "Clasificación por edades",
            ratings,
            index=rating_default,
            help="Es la clasificación ESRB disponible antes del lanzamiento; no es la nota de crítica.",
        )
        c5, c6 = st.columns(2)
        publisher = c5.text_input(
            "Editorial",
            placeholder="Ej. Nintendo",
            help="Puedes dejarlo vacío si aún no se conoce.",
        )
        developer = c6.text_input(
            "Desarrollador",
            placeholder="Ej. Nintendo EAD",
            help="Puedes dejarlo vacío si aún no se conoce.",
        )
        enviar = st.form_submit_button(
            "Calcular estimación",
            type="primary",
            width="stretch",
        )

    if enviar:
        fila = {
            "year": year,
            "platform": platform,
            "genre": genre,
            "rating": rating,
            "publisher": publisher.strip() or np.nan,
            "developer": developer.strip() or np.nan,
        }
        entrada = pd.DataFrame([{columna: fila[columna] for columna in columnas}])
        probabilidad = float(cargar_modelo().predict_proba(entrada)[0, 1])
        with st.container(border=True):
            st.markdown("#### Resultado de la estimación")
            st.metric("Probabilidad de éxito", f"{probabilidad:.1%}")
            st.progress(probabilidad)
            st.caption(
                "La barra representa la probabilidad calculada por el modelo; "
                "no es una garantía ni una clasificación definitiva."
            )
        with st.expander("¿Cómo debo interpretar este resultado?"):
            st.write(
                "El modelo aprendió de videojuegos históricos. La estimación "
                "puede cambiar si cambian las condiciones del mercado y no "
                "demuestra que los datos ingresados causen el resultado."
            )
            st.caption(
                "No se solicitan ventas ni puntuaciones de reseñas porque no "
                "están disponibles antes del lanzamiento."
            )


def formulario_lote(metricas):
    columnas = metricas["predictores"]
    st.write(
        "Sube un CSV con estas columnas. Las columnas adicionales se conservan "
        "en el archivo de salida, pero no se usan para calcular la probabilidad."
    )
    st.code(", ".join(columnas), language="text")
    archivo = st.file_uploader("Selecciona un CSV", type=["csv"], key="lote_csv")
    if archivo is None:
        return

    try:
        datos = pd.read_csv(archivo)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"No se pudo leer el CSV: {error}")
        return

    faltantes = [columna for columna in columnas if columna not in datos.columns]
    if faltantes:
        st.error(f"Faltan columnas requeridas: {', '.join(faltantes)}.")
        return
    anios = pd.to_numeric(datos["year"], errors="coerce")
    anios_invalidos = datos["year"].notna() & anios.isna()
    if anios_invalidos.any():
        st.error(
            f"La columna year contiene {int(anios_invalidos.sum())} valores "
            "no numéricos. Corrígelos antes de continuar."
        )
        return

    resultado = datos.copy()
    resultado["year"] = anios
    resultado["probabilidad_exito"] = cargar_modelo().predict_proba(
        resultado[columnas]
    )[:, 1]
    resultado["prediccion_umbral_0_5"] = (
        resultado["probabilidad_exito"] >= 0.5
    ).astype(int)
    st.success(f"Listo: se analizaron {len(resultado):,} filas.")
    st.dataframe(resultado.head(100), width="stretch")
    st.download_button(
        "Descargar predicciones CSV",
        data=resultado.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicciones_videojuegos.csv",
        mime="text/csv",
    )
    st.caption(
        "El archivo se procesa en memoria durante esta sesión; la aplicación "
        "no lo guarda como un registro permanente."
    )


def pagina_prediccion(metricas):
    encabezado_pagina(
        "Herramienta principal",
        "Estimar el éxito de un videojuego",
        "Calcula una probabilidad antes del lanzamiento. Puedes evaluar un juego "
        "o varios registros desde un archivo CSV.",
    )
    st.info(
        "El objetivo es superar **1 millón de ventas globales**. El modelo no "
        "usa ventas ni reseñas posteriores al lanzamiento."
    )
    dataset = cargar_csv(str(RUTA_DATASET))
    individual, lote = st.tabs(["Un videojuego", "Cargar un CSV"])
    with individual:
        formulario_prediccion(metricas, dataset)
    with lote:
        formulario_lote(metricas)


def pagina_datos():
    encabezado_pagina(
        "Exploración",
        "Conoce los datos",
        "Filtra el catálogo integrado y descarga las fuentes y los resultados "
        "que alimentan los análisis de la aplicación.",
    )
    dataset = cargar_csv(str(RUTA_DATASET))
    anios = pd.to_numeric(dataset["year"], errors="coerce")
    a, b, c = st.columns(3)
    a.metric("Registros", f"{len(dataset):,}")
    b.metric("Variables", f"{dataset.shape[1]:,}")
    c.metric("Éxitos históricos", f"{int(dataset['exito'].sum()):,}")
    st.caption(
        "El catálogo integra ventas y recepción. Las variables de recepción "
        "son retrospectivas y no todas se usan en la predicción pre-lanzamiento."
    )

    explorar, descargas = st.tabs(["Explorar el catálogo", "Descargar archivos"])
    with explorar:
        f1, f2, f3 = st.columns(3)
        generos = ["Todos"] + opciones_categoria(dataset, "genre")
        plataformas = ["Todas"] + opciones_categoria(dataset, "platform")
        genero = f1.selectbox("Filtrar por género", generos)
        plataforma = f2.selectbox("Filtrar por plataforma", plataformas)
        rango_anios = f3.slider(
            "Rango de años",
            min_value=int(anios.min()),
            max_value=int(anios.max()),
            value=(int(anios.min()), int(anios.max())),
        )
        mascara = anios.between(rango_anios[0], rango_anios[1])
        if genero != "Todos":
            mascara &= dataset["genre"].eq(genero)
        if plataforma != "Todas":
            mascara &= dataset["platform"].eq(plataforma)
        filtrados = dataset.loc[mascara]
        st.caption(
            f"{len(filtrados):,} registros coinciden con los filtros. "
            "Se muestran hasta 500 para facilitar la exploración."
        )
        st.dataframe(filtrados.head(500), width="stretch", hide_index=True)
        st.download_button(
            "Descargar selección filtrada",
            data=filtrados.to_csv(index=False).encode("utf-8-sig"),
            file_name="videojuegos_filtrados.csv",
            mime="text/csv",
        )

    with descargas:
        archivos = sorted(
            ruta
            for carpeta in (DATOS, PROCESADOS, IMAGENES)
            for ruta in carpeta.glob("*")
            if ruta.is_file()
            and ruta.suffix.lower() in {".csv", ".json", ".png", ".joblib"}
        )
        if not archivos:
            st.warning("No hay archivos disponibles para descargar.")
            return
        st.write(
            "Elige un archivo para descargar. La lista incluye fuentes originales, "
            "tablas procesadas, métricas, gráficos y modelos entrenados."
        )
        opciones = {
            str(ruta.relative_to(RAIZ)): ruta
            for ruta in archivos
        }
        seleccion = st.selectbox("Archivo", list(opciones))
        ruta = opciones[seleccion]
        tamano_mb = ruta.stat().st_size / (1024 * 1024)
        st.caption(f"Tamaño: {tamano_mb:.2f} MB")
        tipo_mime = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
        st.download_button(
            f"Descargar {ruta.name}",
            data=ruta.read_bytes(),
            file_name=ruta.name,
            mime=tipo_mime,
        )


def pagina_agrupamiento():
    encabezado_pagina(
        "Análisis exploratorio · no predice éxito",
        "Encontrar perfiles de juegos",
        "K-means y Ward agrupan títulos con información incompleta para "
        "explorar similitudes. Los grupos no son categorías verdaderas.",
    )
    comparacion = cargar_csv(
        str(PROCESADOS / "comparacion_modelos_no_supervisados.csv")
    )
    resumen = cargar_csv(str(PROCESADOS / "resumen_clusters.csv"))
    evaluacion = cargar_csv(str(PROCESADOS / "evaluacion_kmeans.csv"))
    asignaciones = cargar_csv(str(PROCESADOS / "juegos_incompletos_clusters.csv"))
    kmeans = comparacion.loc[comparacion["modelo"].eq("K-means")].iloc[0]
    ward = comparacion.loc[
        comparacion["modelo"].eq("Árbol jerárquico (Ward)")
    ].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Casos",
        f"{len(asignaciones):,}",
        help="Juegos con clasificación sin resolver o recepción incompleta.",
    )
    c2.metric(
        "Grupos",
        f"{int(kmeans['k'])}",
        help="Cantidad seleccionada por K-means.",
    )
    c3.metric(
        "ARI",
        f"{ward['ari_vs_kmeans']:.3f}",
        help="Acuerdo entre las asignaciones de K-means y Ward.",
    )

    st.info(
        "La silueta resume qué tan compactos y separados quedan los grupos. "
        "El ARI compara cuánto coinciden K-means y Ward; ninguno es una "
        "medida de predicción del éxito."
    )
    comparacion_tab, perfiles_tab, juegos_tab = st.tabs(
        ["Comparar métodos", "Perfiles de K-means", "Explorar juegos"]
    )
    with comparacion_tab:
        izquierda, derecha = st.columns([1, 1.15])
        with izquierda:
            st.markdown("#### Silueta en la misma muestra")
            st.dataframe(
                comparacion[["modelo", "filas_ajuste", "silueta_muestra"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Comparación en {int(kmeans['filas_ajuste']):,} juegos. "
                "Una silueta cercana a 0 indica separación limitada."
            )
            st.markdown("#### Elección de k")
            st.line_chart(evaluacion.set_index("k")["silueta"])
        with derecha:
            imagen = PROCESADOS / "dendrograma_juegos_incompletos.png"
            if imagen.exists():
                st.image(
                    str(imagen),
                    caption="Árbol de fusiones Ward en una muestra de 1.000 juegos.",
                    width="stretch",
                )
            st.caption(
                "Ward se ejecutó sobre una muestra por el costo computacional "
                "del agrupamiento jerárquico."
            )

    with perfiles_tab:
        st.markdown("#### Resumen de los seis perfiles")
        st.dataframe(resumen, width="stretch", hide_index=True)
        st.caption(
            "La tasa de éxito se calculó después de agrupar y solo describe "
            "el historial; no se usó para construir los grupos."
        )

    with juegos_tab:
        cluster = st.selectbox(
            "Elige un grupo",
            sorted(asignaciones["cluster_kmeans"].dropna().unique().tolist()),
            format_func=lambda valor: f"Cluster {int(valor)}",
        )
        columnas = [
            columna
            for columna in (
                "name", "platform", "year", "genre", "rating",
                "motivo_inclusion", "cluster_kmeans",
            )
            if columna in asignaciones.columns
        ]
        filtradas = asignaciones.loc[
            asignaciones["cluster_kmeans"] == cluster, columnas
        ]
        st.caption(
            f"{len(filtradas):,} juegos pertenecen al grupo seleccionado; "
            "se muestran hasta 300."
        )
        st.dataframe(filtradas.head(300), width="stretch", hide_index=True)
        st.download_button(
            "Descargar juegos de este grupo",
            data=filtradas.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cluster_{int(cluster)}_videojuegos.csv",
            mime="text/csv",
        )


def pagina_metodos_numericos():
    encabezado_pagina(
        "Complementos académicos",
        "Métodos numéricos",
        "Dos ejercicios sobre datos históricos. Ninguno se usa para calcular "
        "la probabilidad pre-lanzamiento.",
    )
    regresion = cargar_json(
        str(PROCESADOS / "metricas_regresion_minimos_cuadrados.json")
    )
    lagrange = cargar_json(
        str(PROCESADOS / "metricas_interpolacion_lagrange.json")
    )
    regresion_tab, lagrange_tab = st.tabs(
        ["Mínimos cuadrados", "Interpolación de Lagrange"]
    )
    with regresion_tab:
        st.markdown("#### Relación histórica: nota de crítica y ventas")
        st.code(regresion["ecuacion"])
        prueba = regresion["prueba_temporal"]
        entrenamiento = regresion["entrenamiento"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² train", f"{entrenamiento['r2']:.3f}")
        c2.metric("R² test", f"{prueba['r2']:.3f}")
        c3.metric("MAE", f"{prueba['mae']:.3f} M")
        c4.metric("RMSE", f"{prueba['rmse']:.3f} M")
        grafico_regresion = PROCESADOS / "regresion_minimos_cuadrados.png"
        if grafico_regresion.exists():
            st.image(str(grafico_regresion), width="stretch")
        st.warning(regresion["advertencia"])
        with st.expander("Cómo leer la ecuación"):
            st.write(
                "El intercepto es el punto de partida de la recta y la pendiente "
                "describe la asociación estimada por cada punto de critic_score. "
                "La relación no prueba causalidad."
            )

    with lagrange_tab:
        st.markdown("#### Validación del promedio anual de ventas")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimación de 2013", f"{lagrange['estimacion']:.4f} M")
        c2.metric(
            "Real 2013",
            f"{lagrange['valor_real_reservado_para_validacion']:.4f} M",
        )
        c3.metric("Error absoluto", f"{lagrange['error_absoluto']:.4f} M")
        st.markdown("**Años usados como nodos**")
        st.dataframe(pd.DataFrame(lagrange["nodos"]), width="stretch", hide_index=True)
        grafico_lagrange = PROCESADOS / "interpolacion_lagrange_ventas_anuales.png"
        if grafico_lagrange.exists():
            st.image(str(grafico_lagrange), width="stretch")
        st.warning(lagrange["advertencia"])


def artefactos_de_fase(clave):
    por_fase = {
        "modelado": [
            "datos_procesados/modelo_exito_pre_lanzamiento.joblib",
            "datos_procesados/metricas_pre_lanzamiento.json",
            "datos_procesados/predicciones_test_pre_lanzamiento.csv",
        ],
        "agrupamiento": [
            "datos_procesados/juegos_incompletos_clusters.csv",
            "datos_procesados/resumen_clusters.csv",
            "datos_procesados/evaluacion_kmeans.csv",
            "datos_procesados/muestra_jerarquica.csv",
            "datos_procesados/comparacion_modelos_no_supervisados.csv",
            "datos_procesados/dendrograma_juegos_incompletos.png",
        ],
        "metodos_numericos": [
            "datos_procesados/regresion_minimos_cuadrados.joblib",
            "datos_procesados/metricas_regresion_minimos_cuadrados.json",
            "datos_procesados/predicciones_regresion_critica_ventas.csv",
            "datos_procesados/regresion_minimos_cuadrados.png",
            "datos_procesados/resultado_interpolacion_lagrange.csv",
            "datos_procesados/metricas_interpolacion_lagrange.json",
            "datos_procesados/interpolacion_lagrange_ventas_anuales.png",
        ],
        "all": [
            str(ruta.relative_to(RAIZ))
            for carpeta in (PROCESADOS, IMAGENES)
            for ruta in sorted(carpeta.glob("*"))
            if ruta.is_file()
        ],
    }
    return [
        str(RAIZ / nombre)
        for nombre in por_fase[clave]
        if (RAIZ / nombre).is_file()
    ]


def pagina_admin():
    encabezado_pagina(
        "Área protegida",
        "Administración",
        "Reentrena modelos o vuelve a ejecutar los análisis. Los artefactos "
        "generados se descargan; no se guardan automáticamente en la nube.",
    )
    st.warning(
        "El reentrenamiento ejecuta scripts de análisis y puede consumir "
        "recursos. El acceso requiere la contraseña privada configurada por "
        "quien administra la aplicación."
    )
    try:
        secreto = st.secrets.get("ADMIN_PASSWORD")
    except StreamlitSecretNotFoundError:
        secreto = None
    if not secreto:
        st.error(
            "Panel deshabilitado: falta configurar ADMIN_PASSWORD en los "
            "secrets de la aplicación."
        )
        return

    if not st.session_state.get("admin_autenticado", False):
        password = st.text_input("Contraseña de administración", type="password")
        if st.button("Iniciar sesión"):
            if hmac.compare_digest(
                password.encode("utf-8"), str(secreto).encode("utf-8")
            ):
                st.session_state["admin_autenticado"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
        return

    if st.button("Cerrar sesión"):
        st.session_state["admin_autenticado"] = False
        st.rerun()

    fases = {
        "Reentrenar el modelo pre-lanzamiento": "modelado",
        "Recalcular K-means y árbol jerárquico": "agrupamiento",
        "Recalcular métodos numéricos": "metodos_numericos",
        "Ejecutar el pipeline CRISP-DM completo": "all",
    }
    etiqueta = st.selectbox("Proceso", list(fases))
    confirmar = st.checkbox("Confirmo que deseo ejecutar el proceso seleccionado.")
    if st.button("Ejecutar", disabled=not confirmar):
        clave = fases[etiqueta]
        argumentos = (
            [sys.executable, str(RAIZ / "main.py"), "--all"]
            if clave == "all"
            else [sys.executable, str(RAIZ / "main.py"), "--fase", clave]
        )
        with st.spinner("Ejecutando el proceso; puede tardar varios minutos..."):
            try:
                resultado = subprocess.run(
                    argumentos,
                    cwd=RAIZ,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                st.error(
                    "El proceso superó el límite de 30 minutos. "
                    "Los archivos temporales de Streamlit Cloud no se guardan "
                    "de forma permanente."
                )
                if error.stdout:
                    st.code(error.stdout[-20000:])
                if error.stderr:
                    st.code(error.stderr[-20000:])
                return

        salida = "\n".join(
            parte for parte in (resultado.stdout, resultado.stderr) if parte
        )
        st.session_state["ultima_ejecucion"] = {
            "codigo": resultado.returncode,
            "salida": salida[-30000:],
            "archivos": (
                artefactos_de_fase(clave) if resultado.returncode == 0 else []
            ),
        }
        st.cache_data.clear()
        st.cache_resource.clear()
        if resultado.returncode == 0:
            st.success("Proceso completado.")
        else:
            st.error(
                f"El proceso terminó con código {resultado.returncode}. "
                "Consulta la salida para identificar el error."
            )

    ejecucion = st.session_state.get("ultima_ejecucion")
    if ejecucion:
        st.subheader("Salida del último proceso")
        st.code(ejecucion["salida"] or "(sin salida)")
        disponibles = [
            ruta for ruta in ejecucion["archivos"] if Path(ruta).is_file()
        ]
        if disponibles:
            st.subheader("Descargar artefactos")
            seleccion = st.selectbox(
                "Artefacto",
                disponibles,
                format_func=lambda ruta: str(Path(ruta).relative_to(RAIZ)),
            )
            ruta = Path(seleccion)
            tipo_mime = (
                mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
            )
            st.download_button(
                f"Descargar {ruta.name}",
                data=ruta.read_bytes(),
                file_name=ruta.name,
                mime=tipo_mime,
            )
        elif ejecucion["codigo"] != 0:
            st.info("No se ofrecen artefactos porque el proceso terminó con error.")
        st.caption(
            "Los archivos generados viven en el almacenamiento efímero de la "
            "aplicación. Descárgalos antes de reiniciar o volver a desplegar; "
            "no se publican ni guardan automáticamente."
        )


def main():
    aplicar_estilos()
    if not RUTA_METRICAS_MODELO.is_file():
        st.error(
            "No se encontró la evaluación del modelo. Incluye los artefactos "
            "de datos_procesados en el despliegue."
        )
        st.stop()
    metricas = cargar_json(str(RUTA_METRICAS_MODELO))

    st.sidebar.markdown(
        """
        <div style="font-size:1.35rem;font-weight:800;color:#fff;padding:.3rem 0 .1rem">
          🎮 Videojuegos
        </div>
        <div style="font-size:.82rem;color:#aab5cb;margin-bottom:1.4rem">
          Minería de datos · CRISP-DM
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("#### Navegación")
    pagina = st.sidebar.radio(
        "Secciones",
        [
            "Inicio",
            "Predicción",
            "Datos y descargas",
            "Segmentación",
            "Métodos numéricos",
            "Administración",
        ],
        label_visibility="collapsed",
        key="navegacion",
    )
    paginas = {
        "Inicio": lambda: mostrar_inicio(metricas),
        "Predicción": lambda: pagina_prediccion(metricas),
        "Datos y descargas": pagina_datos,
        "Segmentación": pagina_agrupamiento,
        "Métodos numéricos": pagina_metodos_numericos,
        "Administración": pagina_admin,
    }
    paginas[pagina]()
    st.sidebar.divider()
    st.sidebar.markdown("#### Objetivo")
    st.sidebar.caption(
        "Estimar si un juego podría superar 1 millón de ventas globales."
    )
    st.sidebar.markdown(
        """
        <div style="font-size:.78rem;color:#aab5cb;line-height:1.5">
          Éxito = ventas globales ≥ 1 millón<br>
          Predictor: solo variables pre-lanzamiento
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
