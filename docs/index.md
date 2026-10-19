# Documentación del proyecto

Documentación técnica y académica del pipeline CRISP-DM.

| Archivo | Contenido |
|---------|-----------|
| [`CATALOGO_FUENTES.md`](CATALOGO_FUENTES.md) | Qué es cada archivo de datos, cada decisión de limpieza y el detalle del cruce |
| [`interpretacion_graficos.md`](interpretacion_graficos.md) | Cómo leer las 9 figuras del EDA: qué dicen y qué no se puede concluir |
| [`Informe_limpieza_union_CRISP_DM.docx`](Informe_limpieza_union_CRISP_DM.docx) | Informe formal de las fases de limpieza e integración |

## Fases del pipeline

1. **Fase 1 · Ingesta** — `src/01_ingesta/auditar_fuentes.py`
2. **Fase 2 · Preparación (ETL)** — `src/02_preparacion/limpieza_datasets.py` y `merged.py`
3. **Fase 3 · EDA** — `src/03_eda/eda_exito.py` (parcial)
4. **Fase 4 · Transformación** — `src/04_transformacion/transformacion.py` (diseño y auditoría de fuga) + `src/04_transformacion/escalamiento.py` (split estratificado, StandardScaler y one-hot)
5. **Fase 5 · Modelado** — `src/05_modelado/modelo_exito.py` (parcial)

Todo se ejecuta desde la raíz con `python main.py --all`.
