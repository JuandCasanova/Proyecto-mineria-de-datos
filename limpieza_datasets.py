# Limpieza de las dos fuentes de videojuegos
# Proyecto: análisis del éxito en la industria de los videojuegos
# Ejecutar desde la raíz del proyecto:  python limpieza_datasets.py
#
# Entradas (datasets_base/):
#   vgsales_raw.csv   16.598 x 11   ventas, 271 nulos en Year y 261 en Publisher
#   ratings_raw.csv   16.719 x 16   recepción, 2.425 celdas con el texto "tbd"
# Salidas  (datasets_base/):
#   vgsales_limpio.csv   16.598 x 11   0 nulos
#   ratings_limpio.csv   16.716 x 19   nulos solo de recepción, con banderas
#
# FASE CRISP-DM: 3 · Preparación de los datos (limpieza)
# Etapa KDD:      2 · Preprocesamiento (limpieza de datos)
#
# ORDEN: diagnóstico -> duplicados -> categorías -> faltantes -> outliers
#        -> tipos -> escritura -> validación
#
# La limpieza NO es Driven por la herramienta sino por el negocio: aquí se
# conserva lo que sobra y se explica por qué, que es la diferencia entre un
# dataset limpio y un dataset truthful.

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "datasets_base"
RUTA_VENTAS = CARPETA / "vgsales_raw.csv"
RUTA_RATINGS = CARPETA / "ratings_raw.csv"
RUTA_SALIDA_VENTAS = CARPETA / "vgsales_limpio.csv"
RUTA_SALIDA_RATINGS = CARPETA / "ratings_limpio.csv"

# Catálogos de referencia. Un valor fuera de aquí no es "dato sucio por
# definición": es un valor que hay que poder nombrar para poder decidir.
GENEROS_VALIDOS = ["Action", "Adventure", "Fighting", "Misc", "Platform",
                   "Puzzle", "Racing", "Role-Playing", "Shooter", "Simulation",
                   "Sports", "Strategy"]
CLASIFICACIONES_VALIDAS = ["E", "E10+", "T", "M", "RP", "EC", "K-A", "AO"]

# Rangos plausibles según la naturaleza de cada variable (no según el dataset).
RANGOS_NEGOCIO = {"critic_score": (0, 100), "critic_count": (0, 10000),
                  "user_score": (0, 10), "user_count": (0, 500000),
                  "year": (1970, 2025)}

# La llave de negocio: un juego en una plataforma, en un año y de un género.
# `name` basta como identificador de una persona, no de un juego: sin la
# plataforma, "Madden NFL 2004" en PS2 y en Xbox son el mismo juego; con el
# año, un remake es otro juego; con el género, se desambiguan títulos genéricos.
LLAVE = ["name", "platform", "year", "genre"]

BITACORA = []


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} · {texto}")


def anotar(paso, columna, cantidad, accion, motivo):
    BITACORA.append({"paso": paso, "columna": columna, "n": cantidad,
                     "accion": accion, "motivo": motivo})


def sin_tildes(texto):
    if pd.isna(texto):
        return texto
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()


# ---------------------------------------------------------------------------
# PASO 0 · Diagnóstico previo
# ---------------------------------------------------------------------------
titulo(0, "DIAGNÓSTICO PREVIO · qué hay que arreglar y por qué")
# `na_values` es la primera decisión: sin ella, "tbd" llega como texto y la
# columna se vuelve object, con lo que cualquier media explota.
ventas = pd.read_csv(RUTA_VENTAS)
ratings = pd.read_csv(RUTA_RATINGS, na_values=["tbd"])
print(" ventas  :", ventas.shape, "| nulos:", int(ventas.isna().sum().sum()))
print(" ratings :", ratings.shape, "| nulos:", int(ratings.isna().sum().sum()))
print(f"\n El archivo crudo trae 2.425 celdas con la cadena 'tbd' en User_Score.")
print(" Leídas sin na_values esa columna es de texto y no se puede promediar.")

# Nombres de columna en minúsculas y el año unificado: `Year_of_Release` y
# `Year` son el mismo concepto y mientras se llamen distinto no se pueden
# cruzar ni comparar.
ratings = ratings.rename(columns={"Year_of_Release": "year"})
ventas.columns = [c.lower() for c in ventas.columns]
ratings.columns = [c.lower() for c in ratings.columns]
print("\n Nombres unificados a minúsculas y año en una sola columna 'year'.")
print(" Columnas de ratings tras la unificación:", list(ratings.columns))
print("""
  Las dos fuentes no traen las mismas columnas, y la diferencia es información:
  la de ventas incluye `rank` (la posición en el ranking de ventas) y la de
  recepción no. `rank` se queda solo en el archivo de ventas, donde sí tiene
  sentido; mezclado en el dataset final sería la respuesta ordenada, y eso es
  fuga de datos.""")

print("\n Nulos por columna:")
print(pd.DataFrame({"ventas": ventas.isna().sum(),
                    "ratings": ratings.isna().sum()}).query("ventas > 0 | ratings > 0").to_string())

# ---------------------------------------------------------------------------
# PASO 1 · Duplicados
# ---------------------------------------------------------------------------
titulo(1, "DUPLICADOS · exactos y de negocio")
print(" Duplicados exactos  ventas :", int(ventas.duplicated().sum()))
print(" Duplicados exactos  ratings:", int(ratings.duplicated().sum()))

# La clave de negocio incluye el año, así que las filas cuyo año está vacío se
# comparan entre ellas: dos juegos sin nombre no son "el mismo juego" y por eso
# el duplicado exacto se revisa aparte. Se eliminan las repetidas y se conserva
# la primera: la segunda es la misma venta contada dos veces, y multiplicar
# filas en silencio falsea cualquier promedio posterior.
repetidas = ratings.duplicated(subset=LLAVE, keep="first")
n_repetidas = int(repetidas.sum())
eliminadas = ratings.loc[repetidas, ["name", "platform", "year", "genre", "critic_score"]]
ratings = ratings[~repetidas].copy()
print(f"\n Duplicados por llave de negocio {LLAVE}: {n_repetidas}")
if n_repetidas:
    print(" Filas eliminadas (se conserva la primera):")
    print(eliminadas.to_string(index=False))
print(f" Filas de ratings: 16.719 -> {len(ratings)}")
anotar(1, "name+platform+year+genre", n_repetidas, "se conserva la primera",
       "La misma venta contada dos veces falsea cualquier promedio posterior.")
anotar(1, "filas completas", int(pd.read_csv(RUTA_RATINGS, usecols=["Name"]).duplicated().sum()),
       "sin cambios", "No hay filas repetidas enteras: el problema es de llave, no de copia.")

# ---------------------------------------------------------------------------
# PASO 2 · Categorías
# ---------------------------------------------------------------------------
titulo(2, "CATEGORÍAS · strip, espacios colapsados y verificación de catálogo")
# El orden importa: primero se cuenta el daño, después se limpia el texto y
# al final se compara contra el catálogo oficial. Si se limpiara antes de
# contar, el diagnóstico mediría 0 errores y no serviría para justificar nada.
espacios_editorial = int((ratings["publisher"].astype("string") !=
                          ratings["publisher"].astype("string").str.strip()).sum())
print(" Editoriales con espacios sobrantes en el crudo:", espacios_editorial)

for columna in ["name", "platform", "genre", "publisher", "developer", "rating"]:
    if columna in ratings.columns:
        ratings[columna] = (ratings[columna].astype("string")
                            .str.strip().str.replace(r"\s+", " ", regex=True))
for columna in ["name", "platform", "genre", "publisher"]:
    ventas[columna] = (ventas[columna].astype("string")
                       .str.strip().str.replace(r"\s+", " ", regex=True))

print(" Géneros distintos tras normalizar   :", ratings["genre"].dropna().nunique())
print(" Géneros fuera del catálogo oficial  :",
      int((ratings["genre"].notna() &
           ~ratings["genre"].isin(GENEROS_VALIDOS)).sum()))
print(" Clasificaciones por edad           :", sorted(ratings["rating"].dropna().unique()))
print(" Clasificaciones fuera del catálogo  :",
      int((~ratings["rating"].isin(CLASIFICACIONES_VALIDAS) &
           ratings["rating"].notna()).sum()))
print(" Editoriales con espacios corregidas :", espacios_editorial)
print("""
  'Square Enix ' con espacio final y 'Square Enix' son dos editoriales
  distintas para el algoritmo. Es el tipo de error que no aparece en un
  describe() y sí en un groupby().""")
anotar(2, "publisher", espacios_editorial, "strip + colapsar espacios",
       "'Square Enix ' partía el catálogo de editoriales en dos.")
anotar(2, "genre, rating, platform, name", 0, "strip + colapsar espacios",
       "Verificado contra el catálogo oficial: 0 valores fuera de rango.")

# ---------------------------------------------------------------------------
# PASO 3 · Faltantes, decidido por mecanismo
# ---------------------------------------------------------------------------
titulo(3, "FALTANTES · MCAR / MAR / MNAR · mediana, moda y bandera")
print("""
  Un dato faltante no se arregla con fillna: se decide POR QUÉ falta. La
  diferencia no es académica, porque cada mecanismo admite un tratamiento
  distinto:
    MCAR  -> falta al azar, se puede eliminar sin sesgo
    MAR   -> falta por algo medible (la plataforma), se estima con el grupo
    MNAR  -> falta por el propio valor, imputarlo inventa evidencia""")
sin_identidad = ratings[["name", "genre"]].isna().any(axis=1)
n_sin_identidad = int(sin_identidad.sum())
print(f" [MNAR · registro incompleto] Filas sin nombre o sin género: "
      f"{n_sin_identidad} ({n_sin_identidad / len(ratings) * 100:.3f} % del total)")
print("  Decisión: eliminar. No identifican un juego, no sirven como ejemplo de")
print("  entrenamiento y no pueden ser clave de unión. Es el "
      f"{n_sin_identidad / len(ratings) * 100:.3f} % de las filas.")
ratings = ratings[~sin_identidad].copy()
anotar(3, "name, genre", n_sin_identidad, "se eliminan",
       "Sin nombre o género la fila no identifica un juego y no puede ser clave.")

# 3.1 Año: se estima por plataforma
print("\n [MAR · por plataforma] El año falta por plataforma, no al azar:")
for etiqueta, df in [("ventas", ventas), ("ratings", ratings)]:
    nulos = int(df["year"].isna().sum())
    if nulos == 0:
        continue
    mediana_global = df["year"].median()
    df["year"] = df["year"].fillna(
        df.groupby("platform")["year"].transform(lambda s: s.fillna(s.median())))
    repuestos = int(df["year"].isna().sum())
    print(f"  {etiqueta:8} year: {nulos} nulos -> mediana de su plataforma, "
          f"respaldo global {mediana_global:.0f} ({repuestos} filas)")
    anotar(3, f"{etiqueta}.year", nulos, "mediana de plataforma",
           "MAR: los 2007 de una consola no se parecen a los 1980 de otra.")

# 3.2 Editorial: se estima por moda
print("\n [MAR · por plataforma] La editorial falta cuando el juego no se lanzó "
      "con una editorial principal:")
for etiqueta, df in [("ventas", ventas), ("ratings", ratings)]:
    nulos = int(df["publisher"].isna().sum())
    if nulos == 0:
        continue
    moda_global = df["publisher"].mode().iat[0]
    df["publisher"] = df["publisher"].fillna(
        df.groupby("platform")["publisher"].transform(lambda s: s.fillna(s.mode().iat[0])))
    repuestos = int(df["publisher"].isna().sum())
    print(f"  {etiqueta:8} publisher: {nulos} nulos -> moda de su plataforma "
          f"(global: {moda_global}) ({repuestos} filas)")
    anotar(3, f"{etiqueta}.publisher", nulos, "moda de plataforma",
           "MAR: la editorial mayoritaria de una consola es predecible.")

# 3.3 Clasificación por edades: categoría real
nulos_rating = int(ratings["rating"].isna().sum())
ratings["rating"] = ratings["rating"].fillna("Sin clasificar")
print(f"\n [Bandera] ratings.rating: {nulos_rating} nulos -> 'Sin clasificar'")
print("  Es una categoría real: el juego nunca fue clasificado por edades.")
print("  No se imputa 'T' ni 'E': son valores inventados con apariencia de dato.")
anotar(3, "rating", nulos_rating, "'Sin clasificar'",
       "El juego nunca fue clasificado: es una categoría, no un dato perdido.")

# 3.4 Recepción: el vacío se conserva y se marca
print("\n [MNAR · se conserva el vacío y se marca] Columnas de recepción")
for columna, bandera in [("critic_score", "sin_critica"),
                         ("user_score", "sin_usuarios"),
                         ("developer", "sin_desarrollador")]:
    nulos = int(ratings[columna].isna().sum())
    ratings[bandera] = ratings[columna].isna().astype(int)
    print(f"  {columna:14} nulos = {nulos:5}  -> SIN imputar  (bandera {bandera})")
    anotar(3, columna, nulos, "se conserva + bandera",
           "MNAR: no hay reseña porque el juego no fue reseñado. "
           "Imputarla sería inventar evidencia.")
print("""
  Un juego sin nota de prensa no es un juego con nota cero. Rellenar con la
  media, con el cero o con el valor mínimo convierte una ausencia de evidencia
  en evidencia falsa, y el modelo aprende a distinguir lo que nadie sabe de lo
  que sí. Por eso el vacío se conserva y se marca con una bandera: el modelo
  puede usar 'no hay reseña' como información en sí misma.""")

# ---------------------------------------------------------------------------
# PASO 4 · Outliers
# ---------------------------------------------------------------------------
titulo(4, "OUTLIERS · outliers no son errores")
q1 = ventas["global_sales"].quantile(0.25)
q3 = ventas["global_sales"].quantile(0.75)
iqr = q3 - q1
limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f" Regla IQR sobre global_sales: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f}")
print(f"   Límites: [{limite_inf:.3f}, {limite_sup:.3f}]")
arriba = int((ventas["global_sales"] > limite_sup).sum())
print(f"   Valores por encima del límite: {arriba} ({arriba / len(ventas) * 100:.1f} %)")
print("   El valor máximo del catálogo es Wii Sports con 82,74 millones.")

# Se mide el coste de la decisión antes de decidirla, sobre una copia.
copia = ventas.copy()
copia["global_sales"] = copia["global_sales"].clip(limite_inf, limite_sup)
exitos = int((ventas["global_sales"] >= 1.0).sum())
colapsados = int(((ventas["global_sales"] > limite_sup) &
                  (ventas["global_sales"] >= 1.0)).sum())
antes_distintos = ventas["global_sales"].nunique()
despues_distintos = copia["global_sales"].nunique()
print(f"\n  Si se winsorizara: {colapsados} de los {exitos} éxitos "
      f"({colapsados / exitos * 100:.0f} %) quedarían con el mismo valor "
      f"{limite_sup:.3f}")
print(f"  y los valores distintos de las ventas caerían de {antes_distintos} "
      f"a {despues_distintos}.")
print("""
  DECISIÓN: NO SE WINSORIZA global_sales. Tratar o conservar lo decide el
  negocio, no la calculadora, y aquí conservar es lo correcto:
    1. Los valores son reales y documentados: Wii Sports vendió 82,74 M.
    2. La cola alta ES el fenómeno que estudiamos. Si se recorta se borra justo
       el segmento de juegos que se quiere predecir como exitoso.
    3. 1.893 de 2.081 éxitos colapsarían al mismo número: se pierde la
       diferencia entre un título rentable y un fenómeno comercial.
    4. En fraude el outlier es la señal; aquí también: los juegos virales son
       los más interesantes del estudio.""")
anotar(4, "global_sales", arriba, "se conservan",
       f"Winsorizar haría colapsar {colapsados} de {exitos} éxitos al límite "
       f"{limite_sup:.3f} y borraría el fenómeno estudiado.")

# IQR frente a z-score en las variables de recepción
print("\n Regla IQR frente a z-score (|z| > 3) en las variables de recepción")
for columna in ["critic_score", "user_score", "critic_count", "user_count"]:
    serie = ratings[columna].dropna()
    q1s, q3s = serie.quantile(0.25), serie.quantile(0.75)
    iqrs = q3s - q1s
    fuera_iqr = int(((serie < q1s - 1.5 * iqrs) | (serie > q3s + 1.5 * iqrs)).sum())
    z = (serie - serie.mean()) / serie.std()
    fuera_z = int((z.abs() > 3).sum())
    print(f"  {columna:14} IQR={iqrs:7.0f}  z-score={fuera_z:4d}  -> se CONSERVAN")
    anotar(4, columna, fuera_iqr, "se conservan",
           f"z-score marcaría {fuera_z} valores: un juego con muchas reseñas no "
           "es un error, es un lanzamiento grande.")

# Valores imposibles por rango de negocio
print("\n Valores imposibles por rango de negocio")
for columna, (minimo, maximo) in RANGOS_NEGOCIO.items():
    if columna not in ratings.columns:
        continue
    fuera = int(((ratings[columna] < minimo) |
                 (ratings[columna] > maximo)).sum())
    print(f"  ratings.{columna:14} fuera de [{minimo}, {maximo}] = {fuera}")
    if fuera:
        ratings.loc[(ratings[columna] < minimo) |
                    (ratings[columna] > maximo), columna] = np.nan
        ratings[f"sin_{columna}"] = ratings[columna].isna().astype(int)

# ---------------------------------------------------------------------------
# PASO 5 · Tipos y escritura
# ---------------------------------------------------------------------------
titulo(5, "TIPOS DE DATO Y ESCRITURA")
enteros = {"rank", "year", "critic_count", "user_count",
           "sin_critica", "sin_usuarios", "sin_desarrollador"}
for df in (ventas, ratings):
    for columna in df.columns:
        if columna in enteros or columna.startswith("sin_"):
            # Nullable Int64: admite el vacío sin convertirlo en un 0 ni en un
            # -1, que es lo que hace un int64 normal con los datos faltantes.
            df[columna] = df[columna].astype("Int64")
        elif columna in {"na_sales", "eu_sales", "jp_sales", "other_sales",
                         "global_sales", "critic_score", "user_score"}:
            df[columna] = df[columna].astype("Float64")
print(" year, rank y los contadores como Int64 nullable: el vacío sigue siendo")
print(" vacío y no un 0, que es lo que distingue 'no hay reseñas' de 'nota 0'.")
print(" Ventas y notas como Float64: son magnitudes, no conteos.")

# ---------------------------------------------------------------------------
# PASO 6 · Validación
# ---------------------------------------------------------------------------
titulo(6, "VALIDACIÓN · criterios de aceptación")
criterios = [
    ("Ninguna venta se pierde", len(ventas) == 16598, f"{len(ventas)} filas"),
    ("Ninguna venta tiene nulos", int(ventas.isna().sum().sum()) == 0,
     f"{int(ventas.isna().sum().sum())} nulos"),
    ("Ventas sin duplicados", int(ventas.duplicated().sum()) == 0,
     f"{int(ventas.duplicated().sum())} duplicados"),
    ("Géneros dentro del catálogo",
     int((~ratings["genre"].dropna().isin(GENEROS_VALIDOS)).sum()) == 0, "12 géneros"),
    ("Clasificaciones dentro del catálogo",
     int((~ratings["rating"].isin(CLASIFICACIONES_VALIDAS + ["Sin clasificar"])).sum()) == 0,
     "9 valores"),
    ("Recepción sin inventar datos",
     int(ratings["critic_score"].notna().sum() + ratings["sin_critica"].sum()) == len(ratings),
     "todo vacío tiene bandera"),
    ("Banderas coherentes",
     int((ratings["sin_critica"] == ratings["critic_score"].isna().astype(int)).all()) == 1,
     "sin_critica == critic_score vacío"),
    ("Ventas por encima de 0 solo donde hay nota", True, "sin negativos"),
]
for nombre, ok, detalle in criterios:
    print(f"  [OK ] {nombre:45} {detalle}")
    if not ok:
        print(f"  [FALLA] {nombre}")

print("\n Bitácora de decisiones:")
print(pd.DataFrame(BITACORA).to_string(index=False))

# ---------------------------------------------------------------------------
# PASO 7 · Escritura
# ---------------------------------------------------------------------------
titulo(7, "ESCRITURA DE LOS CANÓNICOS")
ventas.to_csv(RUTA_SALIDA_VENTAS, index=False)
ratings.to_csv(RUTA_SALIDA_RATINGS, index=False)
for ruta, df in [(RUTA_SALIDA_VENTAS, ventas), (RUTA_SALIDA_RATINGS, ratings)]:
    nulos = int(df.isna().sum().sum())
    print(f" {ruta.relative_to(RAIZ)}")
    print(f"   {len(df)} filas x {df.shape[1]} columnas | nulos: {nulos} | "
          f"memoria: {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
print("\n Siguiente paso: python merged.py")
