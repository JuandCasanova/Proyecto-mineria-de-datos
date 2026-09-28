# Catálogo de fuentes de datos

Proyecto: predicción del éxito comercial de videojuegos con técnicas de minería
de datos (CRISP-DM). Este documento fija **qué es cada archivo, de dónde vino,
qué se le hizo y qué queda después**. Es la referencia para no volver a perder
la trazabilidad de los datos.

---

## 1. Fuentes originales

| # | Fuente | Origen real | Archivo base | Filas x columnas |
|---|--------|-------------|--------------|------------------|
| F1 | Ventas de videojuegos | Dataset público de análisis de ventas (vgsales) | `datasets_base/vgsales_raw.csv` | 16.598 x 11 |
| F2 | Recepción: prensa y usuarios | `mineria_de_datos/archive/Raw Data.csv` | `datasets_base/ratings_raw.csv` | 16.719 x 16 |

### F1 · `vgsales_raw.csv`
Columnas: `Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales,
JP_Sales, Other_Sales, Global_Sales`. Las ventas están en **millones de
unidades**. No tiene nulos en las ventas; sí en `Year` (271) y `Publisher` (261).
Sin duplicados exactos.

> Nota de trazabilidad: el archivo de la raíz `vgsales_cleaned.csv` tenía las
> ventas pero con nombres de columna en minúsculas y `Genre` vacío. Se
> reconstruyó el crudo con los nombres originales a partir de ese archivo
> porque es la única copia de las ventas que existe en el proyecto. Los nulos se
> conservaron: un archivo "crudo" que ya viene limpio no es un crudo, y perder
> la información de qué estaba vacío impediría justificar las imputaciones.

### F2 · `ratings_raw.csv`
Columnas: `Rank, Name, Platform, Year_of_Release, Genre, Publisher, NA_Sales,
EU_Sales, JP_Sales, Other_Sales, Global_Sales, Critic_Score, Critic_Count,
User_Score, User_Count, Developer, Rating`.

Trampas del archivo, y por qué el script de carga las resuelve:

| Trampa | Detalle | Tratamiento |
|--------|---------|-------------|
| `User_Score` es texto | 2.425 celdas con la cadena literal `"tbd"` | `na_values=["tbd"]` y conversión a numérica |
| Nulos de reception | `Critic_Score` 8.582, `User_Score` 9.129 (contando `"tbd"`) | Se conservan con bandera `sin_*` (ver sección 4) |
| Nulos de identidad | `Name` 2, `Genre` 2 | 1 fila se elimina por no identificar un juego |
| Duplicados | 2 filas repeten la llave de negocio | Se conserva la primera |

---

## 2. Pipeline

Las dos etapas son reproducibles: se ejecutan en orden desde la raíz del
proyecto y regeneran los tres archivos derivados **byte a byte** (verificado
con SHA-256).

```
python limpieza_datasets.py     crudos -> vgsales_limpio.csv, ratings_limpio.csv
python merged.py                limpios -> videojuegos_integrado.csv
```

La limpieza aplica, en este orden: diagnóstico, duplicados (por llave de
negocio `name + platform + year + genre`), normalización de categorías contra
catálogo, faltantes decididos por mecanismo (MCAR/MAR/MNAR), outliers medidos
con IQR frente a z-score, tipos con `Int64`/`Float64` nullable, y una
validación con criterios de aceptación y bitácora de decisiones.

Ambas entradas y las dos salidas intermedias se conservan, así que cualquier
decisión de la tabla de la sección 4 puede recalcularse desde el crudo.

---

## 3. Archivos del proyecto

| Archivo | Filas x col. | Nulos | Qué es |
|---------|--------------|-------|--------|
| `datasets_base/vgsales_raw.csv` | 16.598 x 11 | 532 | F1 cruda |
| `datasets_base/ratings_raw.csv` | 16.719 x 16 | 46.716 | F2 cruda |
| `datasets_base/vgsales_limpio.csv` | 16.598 x 11 | 0 | F1 limpia |
| `datasets_base/ratings_limpio.csv` | 16.716 x 19 | 42.035 | F2 limpia (nulos solo de recepción) |
| `datasets_base/videojuegos_integrado.csv` | 16.598 x 23 | 42.341 | **F1 + F2 unidas** |

Nulos del dataset integrado, todos heredados de la fuente de recepción y
conservados a propósito: `critic_score` y `critic_count` 8.564, `user_score` y
`user_count` 9.114, `developer` 6.639, y 62 en `rating` junto con las banderas
`sin_*` (son los 62 juegos que no encontraron ficha de recepción).

---

## 4. Decisiones de limpieza y su justificación

| Decisión | Cantidad | Por qué |
|----------|----------|---------|
| Duplicados por llave de negocio en F2 | 2 filas | Mismo juego, misma plataforma, mismo año y mismo género |
| Filas sin `name` o `genre` | 1 fila | No identifican un juego: inútiles como ejemplo y como clave |
| `year` imputado por mediana de plataforma | 271 (F1) + 269 (F2) | MNAR por plataforma: los 2007 de PS2 no se parecen a los 1980 de Atari |
| `publisher` imputado por moda de plataforma | 261 (F1) + 54 (F2) | MNAR: la editorial mayoritaria de una consola es predecible |
| `rating` ausente -> `"Sin clasificar"` | 6.767 | Es una categoría real: el juego nunca fue clasificado |
| `critic_score`, `critic_count`, `user_score`, `user_count` | 42.035 nulos **conservados** | Imputar una reseña que no existe inventa evidencia. Se marcan con `sin_critica`, `sin_usuarios`, `sin_desarrollador` |
| Outliers de `global_sales` | 1.893 sobre el límite IQR **conservados** | La cola alta es el fenómeno estudiado. Winsorizar haría colapsar 1.893 de los 2.081 éxitos al mismo valor y borraría la diferencia entre 1,1 M y 82 M |

---

## 5. La integración

Clave de cruce: `name_clave + platform + year + genre_clave`, donde `name_clave`
es el nombre normalizado (sin tildes, sin signos, minúsculas) y `genre_clave`
aplica la misma normalización al género.

Resultado: **16.536 de 16.598 ventas encuentran ficha de recepción (99,6 %)** y
62 quedan sin ella (juegos que nadie reseñó, sobre todo de PC y de 2009).

La clave completa no se elige por cobertura sino por integridad: unir solo por
`name` también "coincide" el 99,9 %, pero infla la tabla a 35.202 filas porque
cada venta se empareja con las reseñas del juego en todas las consolas. Medir
el fan-out es parte del trabajo, no un detalle.

Relación de cardinalidad: `many_to_one` (2 filas de ventas comparten juego y
plataforma; hay una sola ficha de recepción). Se verificó con
`validate="many_to_one"`, que corta el proceso si esa cardinalidad se rompe.

---

## 6. La variable objetivo

```
exito = 1  si  global_sales >= 1,0 millones de unidades
```

| Conjunto | Filas | Éxitos | Proporción |
|----------|-------|--------|------------|
| Catálogo completo | 16.598 | 2.081 | 12,5 % |
| Con señal de recepción completa | 6.920 | 1.330 | 19,2 % |
| Entrenamiento | 5.536 | 1.064 | 19,2 % |
| Prueba | 1.384 | 266 | 19,2 % |

El corte de 1,0 millones es una decisión de negocio declarada, no un umbral
estadístico: representa el punto en que un juego deja de ser un proyecto lateral
y pasa a financiar una secuela. Cambiar el umbral cambia el problema, así que
va escrito en el código y no escondido en un parámetro.

**Fuga de datos:** `global_sales`, `na_sales`, `eu_sales`, `jp_sales`,
`other_sales` y `rank` nunca entran en `X`. `rank` además es literalmente la
posición en un ranking de ventas: es la respuesta ordenada.

---

## 7. Variables del modelo (propuesta, pendiente de implementar)

El dataset integrado ya tiene las columnas necesarias; falta el script que
arme la matriz de predictores. La propuesta, con su justificación:

| Tipo | Variables | Tratamiento |
|------|-----------|-------------|
| Numéricas | `year`, `critic_score`, `critic_count`, `user_score`, `user_count` | `StandardScaler` ajustado **solo** con entrenamiento |
| Categóricas nominales | `genre` (12), `platform` (17), `rating` agrupado (4) | One-hot, sin escalar |
| Ordinal genuina | década del juego | Mapa escrito a mano (1980->0 ... 2020->4) |
| Derivada | segmento de `critic_score` | Cuartiles: Mala, Regular, Buena, Excelente |

Descartadas y por qué: `name`, `name_clave`, `genre_clave` son
identificadores; `developer` y `publisher` son texto libre de alta cardinalidad;
`sin_critica`, `sin_usuarios`, `sin_desarrollador` son constantes 0 si se
filtran los casos completos; `exito` es la respuesta.

Dos decisiones que ya están medidas y conviene no repetir al revés:

- **Escalador: `StandardScaler`.** `user_count` tiene asimetría 8,87 y el
  `RobustScaler` la empeora (valor máximo escalado 136,4 frente a 17,9 con
  `StandardScaler`), porque dividir por un IQR pequeño amplifica la cola en vez
  de controlarla.
- **Filtrar a los 6.920 juegos con nota de prensa y de usuarios a la vez**
  (41,7 % del catálogo) sube la tasa de éxito de 12,5 % a 19,2 %. Es un precio
  consciente: sin esa nota, un modelo sería evaluado contra una línea base
  equivocada.

---

## 8. Hallazgos que el dataset integrado ya muestra

Todas las cifras se calculan sobre `datasets_base/videojuegos_integrado.csv`.

1. **La nota de la prensa es el predictor más fuerte.** Tasa de éxito por
   segmento de `critic_score` (cuartiles, sobre los 8.034 juegos que sí tienen
   nota de prensa):

   | Segmento | Juegos | Tasa de éxito |
   |----------|--------|---------------|
   | Mala | 2.060 | 4,1 % |
   | Regular | 2.173 | 8,7 % |
   | Buena | 1.817 | 16,7 % |
   | Excelente | 1.984 | 39,3 % |

   Un juego con prensa excelente tiene **9,6 veces** más probabilidad de superar
   el millón que uno con prensa mala. La señal más fuerte del dataset no es el
   género ni la consola: es lo que dijeron los críticos.

2. **La cobertura de la fuente de recepción es el techo del proyecto.** 8.564
   juegos (el 51,6 % del catálogo) no tienen nota de prensa, y 9.114 no tienen
   nota de usuarios. Cualquier modelo que use esas señales solo podrá aprender
   del 48,4 % del catálogo, y eso se decide en la integración, no en el
   algoritmo.

3. **El género separa, y el orden es contraintuitivo.** En el catálogo
   completo la tasa de éxito va del 22,0 % de `Platform` y 19,2 % de `Shooter`
   hasta el 3,3 % de `Adventure` y 4,7 % de `Strategy`: casi siete veces de
   diferencia entre el mejor y el peor género. Los ganadores son los géneros de
   producción masiva y público amplio; los perdedores, los de diseño propio. La
   lectura de negocio es que el género no explica tanto el éxito como el tipo de
   producción que ese género sugiere en cada estudio.

4. **El conjunto está desbalanceado.** 2.081 éxitos de 16.598 (12,5 %). Un
   modelo que siempre responda "no tiene éxito" acertaría el 87,5 %, así que la
   exactitud sola no sirve para evaluar: hay que usar recall, F1 y matriz de
   confusión.

---

## 9. Referencias

- `merged.py`: script de integración, reproducible desde la raíz del proyecto.
- `Informe_limpieza_union_CRISP_DM.docx`: informe de la entrega anterior, que
  documenta la limpieza y describe el pipeline de cuatro sesiones que se
  descartó. Se conserva como antecedente; el pipeline actual lo sustituye con un
  único script de integración.
