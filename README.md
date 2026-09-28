# Análisis del Éxito en la Industria de los Videojuegos

Proyecto de minería de datos que responde una pregunta de negocio: **¿qué hace
que un videojuego supere el millón de unidades vendidas?** Se construye con dos
fuentes distintas —ventas y recepción (prensa y usuarios)— que se integran en
un único dataset de 16.598 juegos.

## Definición de éxito

```
exito = 1   si   global_sales >= 1,0 millones de unidades
exito = 0   en cualquier otro caso
```

El umbral de 1,0 millones es una decisión de negocio declarada, no un umbral
estadístico: es el punto en que un juego deja de ser un proyecto lateral y pasa
a financiar una secuela. Está escrito en el código y no escondido en un
parámetro, porque cambiarlo cambia el problema.

## Estructura del proyecto

```
Proyecto-mineria-de-datos/
├── datasets_base/                     todas las fuentes y el dataset integrado
│   ├── vgsales_raw.csv                16.598 x 11   ventas (crudo)
│   ├── ratings_raw.csv                16.719 x 16   recepción (crudo)
│   ├── vgsales_limpio.csv             16.598 x 11   ventas, 0 nulos
│   ├── ratings_limpio.csv             16.716 x 19   recepción, nulos solo de reseñas
│   ├── videojuegos_integrado.csv      16.598 x 23   LAS DOS FUENTES EN UNA
│   └── CATALOGO_FUENTES.md            qué es cada archivo y por qué se hizo así
├── limpieza_datasets.py               script de limpieza
├── merged.py                          script de integración
├── README.md
├── Informe_limpieza_union_CRISP_DM.docx
└── Rubrica de evaluación Informe 1.pdf
```

## Cómo ejecutarlo

```bash
python limpieza_datasets.py    # crudos -> los dos canónicos limpios
python merged.py               # limpios -> el dataset integrado
```

Ambas etapas son reproducibles y se validan solas: al ejecutarlas de nuevo
regeneran los archivos **byte a byte** (comprobado con SHA-256). Requisitos:
Python 3.10 o superior con `pandas` y `numpy`.

## La limpieza en tres decisiones

| Decisión | Cantidad | Por qué |
|----------|----------|---------|
| Duplicados por llave de negocio | 2 filas | La misma venta contada dos veces falsea cualquier promedio |
| `year` imputado por mediana de plataforma | 271 + 269 | MAR: el año falta por plataforma, no al azar |
| `publisher` imputado por moda de plataforma | 261 + 54 | MAR: la editorial mayoritaria de una consola es predecible |
| `rating` ausente -> `"Sin clasificar"` | 6.767 | Es una categoría real: el juego nunca fue clasificado |
| Recepción ausente | 42.035 nulos **conservados** | MNAR: no hay reseña porque nadie reseñó. Imputarla sería inventar evidencia. Se marca con `sin_critica`, `sin_usuarios`, `sin_desarrollador` |
| Outliers de `global_sales` | 1.893 **conservados** | Winsorizar haría colapsar 1.893 de los 2.081 éxitos al valor 1,085 y borraría la diferencia entre 1,1 M y 82 M |

## La mezcla en detalle

| Clave de cruce | Coincide | Filas del cruce |
|----------------|----------|-----------------|
| solo `name` | 99,9 % | **35.202** (se infla) |
| `name` + `year` + `genre` | 99,8 % | 31.776 (se infla) |
| `name` + `platform` + `year` + `genre` | 99,6 % | 16.598 |
| **las cuatro, con nombre normalizado** | **99,6 %** | **16.598** |

Se eligen las cuatro claves normalizadas (`name_clave + platform + year +
genre_clave`) **por integridad, no por cobertura**. Unir solo por el nombre
"cubre" el 99,9 %, pero empareja cada venta con las reseñas de ese juego en todas
las consolas e infla la tabla a 35.202 filas. Cobertura alta sin unicidad es
dato inflado.

Resultado: **16.536 de 16.598 ventas encuentran ficha de recepción (99,6 %)**,
0 filas duplicadas, 0 ventas alteradas, cardinalidad verificada con
`validate="many_to_one"`. De las 16.536, 8.034 tienen nota de prensa (48,4 %) y
6.920 tienen nota de prensa y de usuarios a la vez (41,7 %).

## Metodología (CRISP-DM)

1. **Entendimiento del negocio** — umbral de 1,0 millones, pregunta de
   negocio y destinatario del informe.
2. **Entendimiento de los datos** — perfilado de las dos fuentes: se detectaron
   2.425 valores `"tbd"` en `user_score` (texto, no número), 8.580 juegos sin
   nota de prensa y 2 duplicados por llave de negocio.
3. **Preparación de los datos** — limpieza e integración. **Fase completa y
   reproducible:** `limpieza_datasets.py` y `merged.py`.
4. **Modelado** — pendiente: clasificación contra la línea base de 87,5 %.
5. **Evaluación** — pendiente: exactitud, recall, F1 y matriz de confusión (el
   conjunto es desbalanceado, la exactitud sola engañaría).
6. **Despliegue** — pendiente: panel o informe de hallazgos.

## Decisiones que conviene no perder de vista

- **Fuga de datos.** `global_sales`, las ventas por región y `rank` no pueden
  entrar en `X`. `rank` es la posición en un ranking de ventas: es la respuesta
  ordenada.
- **Nulos de recepción no se imputan.** 8.564 juegos no tienen nota de prensa.
  Imputarla sería inventar evidencia; se conservan con banderas `sin_*`.
- **Outliers de ventas se conservan.** La cola alta es el fenómeno estudiado:
  winsorizar haría colapsar 1.893 de los 2.081 éxitos al mismo valor.
- **Las columnas repetidas no se duplican.** Las dos fuentes comparten `name`,
  `platform`, `year`, `genre`, `publisher` y las cinco ventas; de la fuente de
  recepción solo se traen las 9 columnas que la de ventas no tiene. Por eso no
  hay columnas `_x`/`_y`.

## Hallazgos que el dataset integrado ya muestra

| Métrica | Valor |
|---------|-------|
| Games en el catálogo | 16.598 |
| Éxitos (>= 1,0 M) | 2.081 (12,5 %) |
| Géneros con mayor tasa de éxito | Platform 22,0 %, Shooter 19,2 % |
| Tasa de éxito con prensa excelente vs mala | 38,2 % vs 5,0 % (7,6 veces) |
| Cobertura de la fuente de recepción | 99,6 % de las ventas, 48,4 % con nota de prensa |

El detalle completo de cada decisión está en
[`datasets_base/CATALOGO_FUENTES.md`](datasets_base/CATALOGO_FUENTES.md).

## Tecnologías

- **Lenguaje:** Python 3, scripts ejecutables (`.py`), sin notebooks.
- **Manipulación:** `pandas`.
- **Entorno de trabajo:** Windows con PowerShell y Python 3.14.
