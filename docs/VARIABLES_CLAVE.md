# Variables clave para predecir el éxito

Separa, en un solo lugar, qué variable se predice, qué variables entran a
predicción y cuáles quedan fuera. Detención del razonamiento en
[`CATALOGO_FUENTES.md`](CATALOGO_FUENTES.md); la implementación vive en
`src/04_transformacion/transformacion.py` (diseño) y
`src/04_transformacion/escalamiento.py` (ejecución).

---

## 1. Variable objetivo

```
exito = 1  si  global_sales >= 1,0 millones de unidades
exito = 0  en otro caso
```

- **Tipo:** binaria.
- **Umbral:** decisión de negocio declarada (`README.md`, `merged.py:26`),
  no un umbral estadístico.
- **Balance:** 2.081 éxitos sobre 16.598 juegos = **12,5 %** (línea base
  para cualquier modelo).

---

## 2. Variables predictoras (matriz X)

65 columnas de entrada, en cuatro grupos según su tratamiento.

### 2.1 Numéricas (5) — imputación por mediana + `StandardScaler`

| Variable | Contenido | Nulos |
|----------|-----------|-------|
| `year` | Año de lanzamiento | 0 |
| `critic_score` | Nota de prensa | sí (MNAR) |
| `critic_count` | Número de críticas | sí (MNAR) |
| `user_score` | Nota de usuarios | sí (MNAR) |
| `user_count` | Número de votantes | sí (MNAR) |

El escalador se ajusta **solo con train** (`escalamiento.py:164-166`).

### 2.2 Categóricas (3) — one-hot, `handle_unknown="ignore"`

| Variable | Niveles |
|----------|---------|
| `platform` | 36 |
| `genre` | 12 |
| `rating` | 10 (clases ESRB) |

### 2.3 Banderas de nulos preexistentes (3) — binarias, sin escalar

`sin_critica`, `sin_usuarios`, `sin_desarrollador` — marcan la ausencia
de la fuente completa de recepción/desarrollo.

### 2.4 Indicadores MNAR (4) — binarias, sin escalar

`falta_critic_score`, `falta_critic_count`, `falta_user_score`,
`falta_user_count` — construidas **antes** de imputar
(`escalamiento.py:143-144`) para conservar el patrón de faltante, no el
valor que la mediana puso en su lugar.

---

## 3. Variables excluidas

### 3.1 Por fuga de datos (la respuesta o derivados de ella)

`global_sales`, `na_sales`, `eu_sales`, `jp_sales`, `other_sales`,
`rank`, `name`, `name_clave`, `genre_clave`, `exito`.

`rank` es la posición en un ranking de ventas: es la respuesta ordenada.
Auditoría en `transformacion.py:29-37`.

### 3.2 Por alta cardinalidad

| Variable | Niveles | Motivo |
|----------|---------|--------|
| `publisher` | 566 | one-hot denso de casi-un-cero |
| `developer` | 1.696 | idem |

Fuera hasta que se justifique con validación un target/frequency encoding
(`transformacion.py:42-45`).

### 3.3 Identificadores

`name`, `name_clave`, `genre_clave` — memorizarlos es fuga, no aprender.

---

## 4. Señal más fuerte

`critic_score`: tasa de éxito del **39,3 %** con prensa excelente contra
**4,1 %** con prensa mala (≈ 9,6×). El género y la consola separan menos
que la nota de los críticos (`CATALOGO_FUENTES.md` §8).

---

## 5. Propuestas pendientes

| Propuesta | Tratamiento | Estado |
|-----------|-------------|--------|
| Década de lanzamiento | Mapa ordinal (1980→0 … 2020→4) | sin implementar |
| Segmento de `critic_score` | Cuartiles: Mala, Regular, Buena, Excelente | sin implementar |
| `publisher` / `developer` | Target/frequency encoding | solo si gana en validación |

---

## 6. Referencias

| Qué | Dónde |
|-----|-------|
| Definición de éxito | `README.md` · `src/02_preparacion/merged.py:26,172` |
| Diseño de X y auditoría de fuga | `src/04_transformacion/transformacion.py:35-53` |
| Preprocesamiento ejecutado | `src/04_transformacion/escalamiento.py:60-65,140-151` |
| Reparto X/y para modelar | `src/05_modelado/modelo_exito.py:33-34,69-74` |
| Catálogo completo de fuentes y variables | `docs/CATALOGO_FUENTES.md` §6-§7 |
| Correlaciones con el objetivo | `src/03_eda/eda_exito.py:195-204` |
