"""
=============================================================================
  ORQUESTADOR MAESTRO DE PIPELINE ANALÍTICO (CRISP-DM)
  Proyecto: Predicción del Éxito de un Videojuego por Salir
=============================================================================
Este orquestador unifica todas las fases del ciclo de vida del proyecto
garantizando la trazabilidad estricta del paso a paso, la reproducibilidad y
la ejecución controlada por fases o end-to-end.

Uso:
  python main.py --help
  python main.py --all                      # Ejecuta el pipeline completo de principio a fin
  python main.py --fase ingesta             # Fase 1: Auditoría de las fuentes crudas
  python main.py --fase etl                 # Fase 2: Limpieza e integración de las dos fuentes
  python main.py --fase eda                 # Fase 3: Análisis Exploratorio de Datos
  python main.py --fase transformacion      # Fase 4: Codificación y escalamiento de características
  python main.py --fase agrupamiento        # Fase 5: K-means y árbol jerárquico
  python main.py --fase modelado            # Fase 5: Entrenamiento y evaluación temporal
  python main.py --fase metodos_numericos   # Extensión: mínimos cuadrados y Lagrange
  python main.py --status                   # Estado del linaje de datos y artefactos
=============================================================================
"""

import sys
import argparse
import time
import subprocess
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent

# Catálogo oficial de fases del pipeline (organizadas modularmente en src/)
FASES = {
    "ingesta": {
        "numero": "Fase 1",
        "nombre": "Comprensión de los Datos & Auditoría de Fuentes Crudas",
        "scripts": ["src/01_ingesta/auditar_fuentes.py"],
        "descripcion": "Inspección de volumen, nulos, 'tbd' y llave de negocio en ventas y recepción."
    },
    "etl": {
        "numero": "Fase 2",
        "nombre": "Preparación e Integración de Datos (ETL)",
        "scripts": ["src/02_preparacion/limpieza_datasets.py",
                    "src/02_preparacion/merged.py"],
        "descripcion": "Limpieza con decisiones de negocio declaradas y cruce por las 4 claves normalizadas."
    },
    "eda": {
        "numero": "Fase 3",
        "nombre": "Análisis Exploratorio de Datos (EDA)",
        "scripts": ["src/03_eda/eda_exito.py"],
        "descripcion": "Balance de la clase, tasas de éxito por género/plataforma y recepción vs éxito."
    },
    "transformacion": {
        "numero": "Fase 4",
        "nombre": "Transformación & Escalamiento de Características",
        "scripts": ["src/04_transformacion/transformacion.py",
                    "src/04_transformacion/escalamiento.py"],
        "descripcion": "Transformación exploratoria; el modelo supervisado construye su propio preprocesador y split temporal."
    },
    "agrupamiento": {
        "numero": "Fase 5",
        "nombre": "Modelado No Supervisado y Segmentación",
        "scripts": ["src/05_modelado/agrupamiento_no_supervisado.py"],
        "descripcion": "K-means y árbol jerárquico para explorar juegos sin clasificación o con reseñas incompletas."
    },
    "modelado": {
        "numero": "Fase 5",
        "nombre": "Modelado Predictivo & Evaluación vs Línea Base",
        "scripts": ["src/05_modelado/modelo_exito.py"],
        "descripcion": "Compara regresión logística y Random Forest con variables pre-lanzamiento; evalúa probabilidades en años posteriores."
    },
    "metodos_numericos": {
        "numero": "Extensión",
        "nombre": "Métodos Numéricos",
        "scripts": [
            "src/06_metodos_numericos/regresion_minimos_cuadrados.py",
            "src/06_metodos_numericos/interpolacion_lagrange.py",
        ],
        "descripcion": "Análisis histórico de ventas por mínimos cuadrados e interpolación local de promedios anuales con Lagrange."
    }
}

ORDEN_FASES = [
    "ingesta", "etl", "eda", "transformacion", "agrupamiento", "modelado",
    "metodos_numericos",
]


def imprimir_banner():
    print("""
=============================================================================
     PREDICCIÓN DEL ÉXITO DE UN VIDEOJUEGO POR SALIR
             ORQUESTADOR MAESTRO DEL PIPELINE (CRISP-DM)
=============================================================================
""")


def ejecutar_script(script_nombre: str) -> bool:
    ruta_script = BASE_DIR / script_nombre
    if not ruta_script.exists():
        print(f"  [ERROR] No se encuentra el script: {script_nombre}")
        return False

    print(f"\n  >>> Ejecutando módulo: {script_nombre} ...")
    inicio = time.time()

    # Ejecuta en el mismo intérprete de python activo
    resultado = subprocess.run([sys.executable, str(ruta_script)], cwd=str(BASE_DIR))
    duracion = time.time() - inicio

    if resultado.returncode == 0:
        print(f"  [OK] {script_nombre} completado en {duracion:.2f}s")
        return True
    else:
        print(f"  [FALLO] Fallo {script_nombre} (Código de salida: {resultado.returncode})")
        return False


def ejecutar_fase(clave_fase: str) -> bool:
    info = FASES.get(clave_fase)
    if not info:
        print(f"[ERROR] Fase desconocida: '{clave_fase}'. Opciones válidas: {list(FASES.keys())}")
        return False

    print("\n" + "=" * 75)
    print(f"  {info['numero']}: {info['nombre'].upper()}")
    print(f"  Propósito: {info['descripcion']}")
    print("=" * 75)

    for script in info["scripts"]:
        exito = ejecutar_script(script)
        if not exito:
            print(f"\n[INTERRUPCIÓN] El pipeline se detuvo por error en: {script}")
            return False
    return True


def ejecutar_pipeline_completo():
    imprimir_banner()
    print("Iniciando ejecución End-to-End del pipeline analítico...\n")
    inicio_total = time.time()
    fases_exitosas = 0

    for clave in ORDEN_FASES:
        exito = ejecutar_fase(clave)
        if not exito:
            print(f"\n[PIPELINE ABORTADO] Falló la ejecución en la fase: {clave}")
            sys.exit(1)
        fases_exitosas += 1

    duracion_total = time.time() - inicio_total
    print("\n" + "=" * 75)
    print("             PIPELINE ANALÍTICO COMPLETADO EXITOSAMENTE")
    print("=" * 75)
    print(f"Total fases ejecutadas : {fases_exitosas} de {len(ORDEN_FASES)}")
    print(f"Tiempo total ejecución : {duracion_total:.2f} segundos")
    print(f"Artefactos generados   : Carpeta 'datos_procesados/' y 'img/'")
    print(f"Documentación          : Carpeta 'docs/'")
    print("=" * 75)


def mostrar_status():
    imprimir_banner()
    print("ESTADO ACTUAL DE ARTEFACTOS Y LINAJE DE DATOS:\n")
    datos = BASE_DIR / "datos"
    procesados = BASE_DIR / "datos_procesados"
    img = BASE_DIR / "img"

    archivos_clave = [
        ("CRUDO ventas (Bronze)", datos / "vgsales_raw.csv"),
        ("CRUDO recepción (Bronze)", datos / "ratings_raw.csv"),
        ("Ventas limpias (Silver)", procesados / "vgsales_limpio.csv"),
        ("Recepción limpia (Silver)", procesados / "ratings_limpio.csv"),
        ("Dataset integrado (Silver)", procesados / "videojuegos_integrado.csv"),
        ("Split train escalado (Gold)", procesados / "train_escalado.csv"),
        ("Split test escalado (Gold)", procesados / "test_escalado.csv"),
        ("Asignaciones K-means", procesados / "juegos_incompletos_clusters.csv"),
        ("Resumen de clusters", procesados / "resumen_clusters.csv"),
        ("Evaluación K-means", procesados / "evaluacion_kmeans.csv"),
        ("Muestra jerárquica", procesados / "muestra_jerarquica.csv"),
        ("Comparación K-means y jerárquico", procesados / "comparacion_modelos_no_supervisados.csv"),
        ("Dendrograma jerárquico", procesados / "dendrograma_juegos_incompletos.png"),
        ("Modelo pre-lanzamiento", procesados / "modelo_exito_pre_lanzamiento.joblib"),
        ("Métricas pre-lanzamiento", procesados / "metricas_pre_lanzamiento.json"),
        ("Predicciones temporales", procesados / "predicciones_test_pre_lanzamiento.csv"),
        ("Modelo mínimos cuadrados", procesados / "regresion_minimos_cuadrados.joblib"),
        ("Métricas mínimos cuadrados", procesados / "metricas_regresion_minimos_cuadrados.json"),
        ("Predicciones de regresión", procesados / "predicciones_regresion_critica_ventas.csv"),
        ("Gráfico mínimos cuadrados", procesados / "regresion_minimos_cuadrados.png"),
        ("Resultado interpolación Lagrange", procesados / "resultado_interpolacion_lagrange.csv"),
        ("Métricas interpolación Lagrange", procesados / "metricas_interpolacion_lagrange.json"),
        ("Gráfico interpolación Lagrange", procesados / "interpolacion_lagrange_ventas_anuales.png"),
        ("fig1 barras balance de clase", img / "fig1_barras_exito.png"),
        ("fig2 barras tasa por género", img / "fig2_barras_genero.png"),
        ("fig3 histograma ventas global", img / "fig3_hist_ventas_global.png"),
        ("fig4 boxplot ventas por género", img / "fig4_box_ventas_genero.png"),
        ("fig5 scatter prensa vs ventas", img / "fig5_scatter_critico_ventas.png"),
        ("fig6 heatmap correlaciones", img / "fig6_heatmap_correlaciones.png"),
        ("fig7 matriz de dispersión", img / "fig7_matriz_dispersion.png"),
        ("fig8 líneas evolución por año", img / "fig8_lineas_evolucion_anio.png"),
        ("fig9 barras top plataformas", img / "fig9_barras_plataforma.png"),
    ]
    for nombre, ruta in archivos_clave:
        estado = "[OK]" if ruta.exists() else "[PENDIENTE]"
        tamano = f"({ruta.stat().st_size / 1024:.1f} KB)" if ruta.exists() else ""
        print(f"  {nombre:32} : {estado} {tamano}")

    print("\nFases pendientes de implementar por completo:")
    for clave, info in FASES.items():
        if clave == "eda":
            print(f"  - {info['numero']} {info['nombre']} (esqueleto con TODOs)")

    print("\nPara ejecutar una fase: python main.py --fase <nombre_fase>")
    print("Para ejecutar todo    : python main.py --all")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador Maestro del Pipeline de Minería de Datos (CRISP-DM)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta todas las fases del pipeline en orden secuencial (End-to-End)"
    )
    parser.add_argument(
        "--fase",
        choices=ORDEN_FASES,
        help="""Ejecuta una fase específica del pipeline:
  ingesta        : Auditoría de las fuentes crudas
  etl            : Limpieza e integración (ventas + recepción)
  eda            : Análisis Exploratorio de Datos
  transformacion : Codificación y escalamiento de características
  agrupamiento   : K-means y árbol jerárquico de casos incompletos
  modelado       : Modelo predictivo pre-lanzamiento y evaluación temporal
  metodos_numericos : Ajuste por mínimos cuadrados e interpolación de Lagrange"""
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Muestra el estado actual del linaje de datos y artefactos generados"
    )

    args = parser.parse_args()

    if args.status:
        mostrar_status()
        return

    if args.all:
        ejecutar_pipeline_completo()
    elif args.fase:
        imprimir_banner()
        exito = ejecutar_fase(args.fase)
        if exito:
            print(f"\n[OK FASE COMPLETADA] La fase '{args.fase}' finalizó con éxito.")
        else:
            sys.exit(1)
    else:
        imprimir_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
