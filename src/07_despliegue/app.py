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
    st.title("¿Qué hace que un videojuego tenga éxito?")
    st.write(
        "Explora los datos históricos de videojuegos, consulta los análisis "
        "y estima la probabilidad de superar un millón de unidades vendidas."
    )
    st.info(
        "La predicción usa únicamente información disponible antes del "
        "lanzamiento. Es una estimación basada en datos históricos, no una "
        "garantía de ventas ni una recomendación comercial."
    )

    division = metricas["division"]
    prueba = metricas["metricas_test"]["modelo"]
    baseline = metricas["metricas_test"]["baseline_dummy_prior"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Juegos usados", f"{division['train_filas'] + division['test_filas']:,}")
    c2.metric("Éxitos en prueba", f"{division['test_tasa_exito']:.1%}")
    c3.metric("Average Precision", f"{prueba['average_precision']:.3f}")

    st.subheader("Evaluación temporal del modelo")
    st.caption(
        f"Entrenamiento hasta {division['train_anio_maximo']}; prueba desde "
        f"{division['anio_inicio_test']}. Random Forest se seleccionó frente "
        "a regresión logística usando validación cruzada."
    )
    a, b, c = st.columns(3)
    a.metric(
        "ROC-AUC",
        f"{prueba['roc_auc']:.3f}",
        help="Capacidad de ordenar casos positivos por encima de los negativos.",
    )
    b.metric(
        "Brier score",
        f"{prueba['brier_score']:.3f}",
        help="Error de las probabilidades; menor es mejor.",
    )
    c.metric(
        "Average Precision (baseline)",
        f"{baseline['average_precision']:.3f}",
        help="Referencia de un clasificador que asigna la tasa histórica de éxito.",
    )
    st.caption(
        f"Al umbral informativo de 0,50, el recall de éxito fue "
        f"{prueba['recall_exito']:.1%}; no detectar todos los éxitos es una "
        "limitación importante del modelo."
    )

    figura = IMAGENES / "fig1_barras_exito.png"
    if figura.exists():
        st.image(str(figura), caption="Distribución histórica de la clase objetivo.")


def opciones_categoria(dataset, columna):
    valores = dataset[columna].dropna().astype(str).str.strip()
    return sorted(valor for valor in valores.unique() if valor)


def formulario_prediccion(metricas, dataset):
    columnas = metricas["predictores"]
    st.subheader("Un videojuego")
    st.caption(
        "Indica datos conocidos antes del lanzamiento. Si no conoces la "
        "editorial o el desarrollador, deja el campo vacío."
    )
    with st.form("formulario_prediccion"):
        c1, c2, c3 = st.columns(3)
        year = c1.number_input(
            "Año de lanzamiento",
            min_value=1980,
            max_value=2035,
            value=2020,
            step=1,
        )
        platform = c2.selectbox("Plataforma", opciones_categoria(dataset, "platform"))
        genre = c3.selectbox("Género", opciones_categoria(dataset, "genre"))
        c4, c5, c6 = st.columns(3)
        ratings = opciones_categoria(dataset, "rating")
        rating_default = (
            ratings.index("Sin clasificar")
            if "Sin clasificar" in ratings
            else 0
        )
        rating = c4.selectbox("Clasificación", ratings, index=rating_default)
        publisher = c5.text_input("Editorial", placeholder="Ej. Nintendo")
        developer = c6.text_input("Desarrollador", placeholder="Ej. Nintendo EAD")
        enviar = st.form_submit_button("Estimar probabilidad")

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
        st.metric("Probabilidad estimada de éxito", f"{probabilidad:.1%}")
        st.progress(probabilidad)
        st.caption(
            "Éxito significa ventas globales históricas de al menos 1 millón. "
            "La cifra no indica causalidad y puede no representar las "
            "condiciones actuales del mercado."
        )


def formulario_lote(metricas):
    columnas = metricas["predictores"]
    st.subheader("Varios videojuegos desde un CSV")
    st.write(
        "El archivo debe tener una columna por cada variable indicada abajo. "
        "Las columnas adicionales se conservan en el resultado, pero no se "
        "usan como entradas del modelo."
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
    st.success(f"Se analizaron {len(resultado):,} filas.")
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
    st.title("Predicción pre-lanzamiento")
    dataset = cargar_csv(str(RUTA_DATASET))
    formulario_prediccion(metricas, dataset)
    st.divider()
    formulario_lote(metricas)


def pagina_datos():
    st.title("Datos y descargas")
    dataset = cargar_csv(str(RUTA_DATASET))
    a, b, c = st.columns(3)
    a.metric("Registros integrados", f"{len(dataset):,}")
    b.metric("Variables", f"{dataset.shape[1]:,}")
    c.metric("Tasa histórica de éxito", f"{dataset['exito'].mean():.1%}")
    st.caption(
        "El dataset integrado combina ventas y recepción de crítica/usuarios. "
        "Los datos y las métricas de recepción son retrospectivos; no todos "
        "se utilizan en la predicción pre-lanzamiento."
    )
    st.dataframe(dataset.head(500), width="stretch")

    archivos = sorted(
        ruta
        for carpeta in (DATOS, PROCESADOS)
        for ruta in carpeta.glob("*")
        if ruta.is_file() and ruta.suffix.lower() in {".csv", ".json", ".png"}
    )
    if not archivos:
        st.warning("No hay archivos de datos disponibles para descargar.")
        return

    st.subheader("Archivos del proyecto")
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
    st.title("Segmentación de casos incompletos")
    st.write(
        "K-means y el árbol jerárquico Ward agrupan juegos sin rating o con "
        "reseñas incompletas. Son métodos exploratorios no supervisados; sus "
        "grupos no son categorías verdaderas ni predicciones causales."
    )
    comparacion = cargar_csv(
        str(PROCESADOS / "comparacion_modelos_no_supervisados.csv")
    )
    st.dataframe(comparacion, width="stretch", hide_index=True)
    st.caption(
        "La silueta resume cohesión y separación (mayor es mejor). El ARI "
        "mide el acuerdo de las particiones en la misma muestra; no hay una "
        "etiqueta verdadera con la cual medir accuracy."
    )

    resumen = cargar_csv(str(PROCESADOS / "resumen_clusters.csv"))
    st.subheader("Perfiles K-means")
    st.dataframe(resumen, width="stretch", hide_index=True)
    st.caption(
        "La tasa de éxito por cluster se calcula después del agrupamiento y "
        "solo describe los datos históricos; no fue usada para formar los grupos."
    )

    evaluacion = cargar_csv(str(PROCESADOS / "evaluacion_kmeans.csv"))
    st.subheader("Selección de k")
    st.line_chart(evaluacion.set_index("k")["silueta"])
    imagen = PROCESADOS / "dendrograma_juegos_incompletos.png"
    if imagen.exists():
        st.image(str(imagen), caption="Dendrograma Ward en una muestra reproducible.")

    asignaciones = cargar_csv(str(PROCESADOS / "juegos_incompletos_clusters.csv"))
    cluster = st.selectbox(
        "Filtrar asignaciones por cluster",
        sorted(asignaciones["cluster_kmeans"].dropna().unique().tolist()),
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
    st.dataframe(filtradas.head(1000), width="stretch", hide_index=True)
    st.caption(f"Mostrando hasta 1.000 de {len(filtradas):,} juegos del cluster.")


def pagina_metodos_numericos():
    st.title("Métodos numéricos")
    st.info(
        "Estos análisis históricos son complementarios y no forman parte del "
        "predictor pre-lanzamiento."
    )
    regresion = cargar_json(
        str(PROCESADOS / "metricas_regresion_minimos_cuadrados.json")
    )
    st.subheader("Mínimos cuadrados: crítica y ventas")
    st.code(regresion["ecuacion"])
    prueba = regresion["prueba_temporal"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² en prueba", f"{prueba['r2']:.3f}")
    c2.metric("MAE", f"{prueba['mae']:.3f} M")
    c3.metric("RMSE", f"{prueba['rmse']:.3f} M")
    st.caption(regresion["advertencia"])
    grafico_regresion = PROCESADOS / "regresion_minimos_cuadrados.png"
    if grafico_regresion.exists():
        st.image(str(grafico_regresion))

    lagrange = cargar_json(
        str(PROCESADOS / "metricas_interpolacion_lagrange.json")
    )
    st.subheader("Interpolación de Lagrange")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimación 2013", f"{lagrange['estimacion']:.4f} M")
    c2.metric(
        "Promedio real reservado",
        f"{lagrange['valor_real_reservado_para_validacion']:.4f} M",
    )
    c3.metric("Error absoluto", f"{lagrange['error_absoluto']:.4f} M")
    st.dataframe(pd.DataFrame(lagrange["nodos"]), width="stretch")
    st.caption(lagrange["advertencia"])
    grafico_lagrange = PROCESADOS / "interpolacion_lagrange_ventas_anuales.png"
    if grafico_lagrange.exists():
        st.image(str(grafico_lagrange))


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
    st.title("Administración del proyecto")
    st.write(
        "El reentrenamiento ejecuta scripts de análisis y puede consumir "
        "recursos. El acceso requiere el secreto ADMIN_PASSWORD configurado "
        "por quien administra la aplicación en Streamlit Cloud."
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
    if not RUTA_METRICAS_MODELO.is_file():
        st.error(
            "No se encontró la evaluación del modelo. Incluye los artefactos "
            "de datos_procesados en el despliegue."
        )
        st.stop()
    metricas = cargar_json(str(RUTA_METRICAS_MODELO))

    st.sidebar.title("Videojuegos y datos")
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
    st.sidebar.caption(
        "Datos históricos del proyecto de minería de datos. "
        "Las predicciones son estimaciones, no garantías."
    )


if __name__ == "__main__":
    main()
