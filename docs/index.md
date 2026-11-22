# Documentación del proyecto

Documentación técnica y académica del pipeline CRISP-DM.

| Archivo | Contenido |
|---------|-----------|
| [`CATALOGO_FUENTES.md`](CATALOGO_FUENTES.md) | Qué es cada archivo de datos, cada decisión de limpieza y el detalle del cruce |
| [`VARIABLES_CLAVE.md`](VARIABLES_CLAVE.md) | Objetivo, variables pre-lanzamiento, exclusiones y evaluación temporal del modelo |
| [`interpretacion_graficos.md`](interpretacion_graficos.md) | Cómo leer las 9 figuras del EDA: qué dicen y qué no se puede concluir |
| [`Informe_limpieza_union_CRISP_DM.docx`](Informe_limpieza_union_CRISP_DM.docx) | Informe formal de las fases de limpieza e integración |

## Fases del pipeline

1. **Fase 1 · Ingesta** — `src/01_ingesta/auditar_fuentes.py`
2. **Fase 2 · Preparación (ETL)** — `src/02_preparacion/limpieza_datasets.py` y `merged.py`
3. **Fase 3 · EDA** — `src/03_eda/eda_exito.py` (parcial)
4. **Fase 4 · Transformación exploratoria** — `src/04_transformacion/transformacion.py` y `escalamiento.py`; el modelo supervisado usa su propio preprocesador y split temporal
5. **Fase 5 · Agrupamiento no supervisado** — `src/05_modelado/agrupamiento_no_supervisado.py` (K-means y árbol jerárquico sobre casos sin clasificación o reseñas incompletas)
6. **Fase 5 · Modelado supervisado** — `src/05_modelado/modelo_exito.py` (comparación, calibración y evaluación temporal de probabilidades pre-lanzamiento)
7. **Extensión · Métodos numéricos** — `src/06_metodos_numericos/` (regresión lineal por mínimos cuadrados e interpolación de Lagrange sobre agregados históricos)

Todo se ejecuta desde la raíz con `python main.py --all`.
