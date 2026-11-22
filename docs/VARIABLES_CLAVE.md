# Variables para predecir el éxito antes del lanzamiento

Este documento describe las variables del modelo supervisado de
`src/05_modelado/modelo_exito.py`. La predicción se realiza sobre videojuegos
históricos, simulando que el modelo se entrena con juegos antiguos y predice
los lanzamientos de años posteriores.

## 1. Objetivo

```text
exito = 1 si global_sales >= 1,0 millones de unidades
exito = 0 en otro caso
```

El umbral es una decisión de negocio y no entra como predictor.

## 2. Variables disponibles antes del lanzamiento

| Variable | Tratamiento | Motivo |
|----------|-------------|--------|
| `year` | Imputación por mediana y `StandardScaler` | Año de lanzamiento planificado |
| `platform` | One-hot | Plataforma prevista |
| `genre` | One-hot | Género |
| `rating` | One-hot; vacío como `"Sin dato"` | Clasificación por edades, si está disponible |
| `publisher` | One-hot; categorías con menos de 20 ejemplos se agrupan | Editorial conocida al anunciar el juego |
| `developer` | One-hot; categorías con menos de 20 ejemplos se agrupan | Desarrollador conocido al anunciar el juego |

`OneHotEncoder` se ajusta dentro del pipeline de cada fold, agrupa niveles
infrecuentes y tolera categorías no vistas. Así se evita ajustar el
preprocesamiento con el conjunto de prueba.

## 3. Variables excluidas

- **Resultado o derivados:** `global_sales`, ventas por región, `rank` y
  `exito`. Las ventas definen el objetivo; `rank` deriva de ellas.
- **Identificadores:** `name`, `name_clave` y `genre_clave`. No deben
  memorizarse como características.
- **Datos posteriores al lanzamiento:** `critic_score`, `critic_count`,
  `user_score`, `user_count`, `sin_critica`, `sin_usuarios` y
  `sin_desarrollador`. No estarían disponibles al predecir un juego que aún
  no se ha publicado.

## 4. Evaluación temporal

El entrenamiento usa juegos con `year <= 2011`; la prueba reserva juegos con
`year >= 2012`. Los candidatos se comparan con cinco folds estratificados
dentro del entrenamiento. La métrica principal para seleccionar el modelo es
Average Precision, porque solo 12,5 % de los juegos históricos son éxitos.
Las probabilidades del candidato ganador se calibran usando únicamente train.

La prueba temporal simula mejor una predicción futura que una partición
aleatoria, aunque el catálogo llega solo hasta 2020 y no garantiza que el
mercado actual tenga la misma distribución.

## 5. Artefactos

- `datos_procesados/modelo_exito_pre_lanzamiento.joblib`: pipeline calibrado.
- `datos_procesados/metricas_pre_lanzamiento.json`: configuración y métricas.
- `datos_procesados/predicciones_test_pre_lanzamiento.csv`: probabilidades y
  resultados reales del periodo de prueba.
