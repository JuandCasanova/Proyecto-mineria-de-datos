# Mezcla de las dos fuentes de videojuegos
# Proyecto: análisis del éxito en la industria de los videojuegos
# Ejecutar desde la raíz del proyecto:  python merged.py
#
# Entradas (datasets_base/):
#   vgsales_limpio.csv   16.598 x 11   ventas, sin nulos
#   ratings_limpio.csv   16.716 x 19   recepción, nulos solo de reseñas
# Salida  (datasets_base/):
#   videojuegos_integrado.csv   16.598 x 23   una fila por juego-venta
#
# FASE CRISP-DM: 3 · Preparación de los datos (integración)
# Etapa KDD:      2 · Preprocesamiento (integración de datos)

from pathlib import Path
import unicodedata

import pandas as pd

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "datasets_base"
RUTA_VENTAS = CARPETA / "vgsales_limpio.csv"
RUTA_RATINGS = CARPETA / "ratings_limpio.csv"
RUTA_SALIDA = CARPETA / "videojuegos_integrado.csv"

UMBRAL_EXITO = 1.0   # millones de unidades

# Columnas que aporta la fuente de recepción y la de ventas no tiene.
# Se traen SOLO estas: las dos fuentes comparten name, platform, year, genre,
# publisher y las cinco ventas, y volver a traerlas generaría columnas _x/_y
# con el mismo dato duplicado dos veces.
COLUMNAS_RECEPCION = ["critic_score", "critic_count", "user_score", "user_count",
                      "developer", "rating",
                      "sin_critica", "sin_usuarios", "sin_desarrollador"]

CLAVES = ["name_clave", "platform", "year", "genre_clave"]


def normalizar(texto):
    """Minúsculas, sin tildes y sin signos: la clave no puede depender del formato."""
    if pd.isna(texto):
        return texto
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(texto.lower().replace(".", " ").replace("-", " ").split())


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} · {texto}")


# ---------------------------------------------------------------------------
# PASO 0 · Cargar las dos fuentes
# ---------------------------------------------------------------------------
titulo(0, "CARGA DE LAS DOS FUENTES")
ventas = pd.read_csv(RUTA_VENTAS)
ratings = pd.read_csv(RUTA_RATINGS)
print(" ventas  :", ventas.shape, "| nulos:", int(ventas.isna().sum().sum()))
print(" ratings :", ratings.shape, "| nulos:", int(ratings.isna().sum().sum()))
print("\n Las dos fuentes miden la misma población de juegos:")
print("   juegos distintos en ventas  :", ventas["name"].nunique())
print("   juegos distintos en ratings :", ratings["name"].nunique())
print("   rango de años   ventas :", (int(ventas["year"].min()), int(ventas["year"].max())))
print("   rango de años   ratings:", (int(ratings["year"].min()), int(ratings["year"].max())))

# ---------------------------------------------------------------------------
# PASO 1 · Construir la clave de cruce
# ---------------------------------------------------------------------------
titulo(1, "CLAVE DE CRUCE · por qué no basta con unir por el nombre")
# Unir por `name` a secas falla con títulos como "Madden NFL 2004" frente a
# "Madden NFL 2004!", o por diferencias de tildes y signos. La clave
# normalizada SOLO se usa para cruzar; el nombre original queda intacto.
for df in (ventas, ratings):
    df["name_clave"] = df["name"].map(normalizar)
    df["genre_clave"] = df["genre"].map(normalizar)

print(" Claves:", CLAVES)
print(" Cada una descarta un error distinto:")
print("   name_clave -> tildes, puntos, guiones y mayúsculas diferentes")
print("   platform   -> el mismo juego en PS2 y en Xbox son dos registros")
print("   year       -> un remake del mismo título es otro juego")
print("   genre_clave-> desambigua títulos genéricos")
ejemplos = ventas.loc[ventas["name"] != ventas["name_clave"], ["name", "name_clave"]]
print("\n Títulos que la normalización unifica (muestra):")
print(ejemplos.head(4).to_string(index=False))

# ---------------------------------------------------------------------------
# PASO 2 · El inconveniente real: cobertura no es integridad
# ---------------------------------------------------------------------------
titulo(2, "MEDICIÓN DEL CRUCE · cobertura contra integridad")
variantes = [
    ("solo name", ["name"]),
    ("name + year + genre", ["name", "year", "genre"]),
    ("name + platform + year + genre", ["name", "platform", "year", "genre"]),
    ("LAS CUATRO CLAVES NORMALIZADAS", CLAVES),
]
print(f"  {'clave de cruce':36} {'coincide':>9} {'%':>7} {'filas del cruce':>17}")
for etiqueta, claves in variantes:
    aux = ratings[claves].drop_duplicates()
    n_coincide = len(ventas[claves].merge(aux, on=claves, how="inner"))
    n_cruce = len(ventas.merge(ratings[claves], on=claves, how="left"))
    print(f"  {etiqueta:36} {n_coincide:9} {n_coincide / len(ventas) * 100:6.1f} % "
          f"{n_cruce:17}")
print(f"  {'(referencia: filas de ventas)':36} {len(ventas):9} {100.0:6.1f} % "
      f"{len(ventas):17}")
print("""
  Aquí está el inconveniente más importante de la mezcla, y no es la cobertura:
  con `solo name` coincide el 99,9 % de las filas, pero el cruce se infla a más
  de 30.000. Cada venta de un juego en PS2 queda emparejada con las reseñas de
  ese juego en TODAS las consolas, y la misma nota se cuenta varias veces.

    Cobertura alta sin unicidad = tabla inflada y pesos falsos en el modelo.
    Cobertura algo menor con 1:1 = tabla confiable.

  Por eso se usan las cuatro claves: son conjuntas y únicas.""")

# ---------------------------------------------------------------------------
# PASO 3 · La unión
# ---------------------------------------------------------------------------
titulo(3, "UNIÓN · left merge con validate='many_to_one'")
recepcion = ratings[COLUMNAS_RECEPCION + CLAVES].drop_duplicates(subset=CLAVES)
print(" Filas de recepción con una sola fila por clave:", len(recepcion))
print(" Duplicados restantes en la clave:", int(recepcion.duplicated(subset=CLAVES).sum()))
repetidas = int(ventas.duplicated(subset=CLAVES).sum())
print(f" Filas de ventas que comparten clave: {repetidas}")
print("""
  how='left' -> la tabla de ventas es la base: ningún juego se pierde porque no
               tenga reseña.
  validate   -> una fila de recepción pertenece como máximo a una venta. La
               relación correcta es muchas-a-una: un mismo juego-plataforma
               puede tener varias filas de ventas, pero una sola ficha. Si esa
               cardinalidad se rompiera, pandas detiene el proceso en vez de
               multiplicar filas en silencio.""")

antes = len(ventas)
integrado = ventas.merge(recepcion, on=CLAVES, how="left",
                         indicator=True, validate="many_to_one")
print(f"\n Filas antes={antes}  después={len(integrado)}  "
      f"(una unión muchas-a-una no cambia el número de filas)")

# ---------------------------------------------------------------------------
# PASO 4 · Validación
# ---------------------------------------------------------------------------
titulo(4, "VALIDACIÓN DEL RESULTADO")
con_match = int((integrado["_merge"] == "both").sum())
sin_match = int((integrado["_merge"] == "left_only").sum())
print(f" Con ficha de recepción : {con_match} ({con_match / len(ventas) * 100:.1f} %)")
print(f" Sin coincidencia      : {sin_match} ({sin_match / len(ventas) * 100:.1f} %)")
print(" Ventas globales intactas:",
      integrado["global_sales"].equals(ventas["global_sales"]))
print(" Filas duplicadas      :", int(integrado.duplicated().sum()))

# El motivo de las que no cruzaron
sin_coincidencia = integrado[integrado["_merge"] == "left_only"]
print(f"\n Las {len(sin_coincidencia)} ventas sin recepción no son un error:")
print(" son juegos que la prensa y los usuarios nunca reseñaron. Conservan NaN")
print(" en las columnas de recepción y las banderas sin_* encendidas.")
print(" Géneros más frecuentes entre las que no cruzaron:")
print(sin_coincidencia["genre_clave"].value_counts().head(3).to_string())
print(" Ejemplos:")
print(sin_coincidencia[["name", "platform", "year", "genre"]].head(4).to_string(index=False))

# Cobertura de la señal de recepción
con_nota = int(integrado["critic_score"].notna().sum())
ambos = int((integrado["critic_score"].notna() & integrado["user_score"].notna()).sum())
print(f"\n Con nota de prensa      : {con_nota} ({con_nota / len(ventas) * 100:.1f} %)")
print(f" Con nota de prensa y de usuarios: {ambos} ({ambos / len(ventas) * 100:.1f} %)")

# ---------------------------------------------------------------------------
# PASO 5 · El objetivo
# ---------------------------------------------------------------------------
titulo(5, "VARIABLE OBJETIVO · 'éxito' definido y separado")
integrado["exito"] = (integrado["global_sales"] >= UMBRAL_EXITO).astype(int)
print(f" exito = 1 si global_sales >= {UMBRAL_EXITO} millones de unidades")
print("\n Distribución del objetivo:")
print(integrado["exito"].value_counts().to_string())
print(f" Proporción de éxitos: {integrado['exito'].mean():.3f}")
print("\n Tasa de éxito por género (top 6):")
print(integrado.groupby("genre")["exito"].agg(["sum", "count", "mean"])
      .sort_values("mean", ascending=False).head(6).round(3).to_string())
print("""
  El conjunto queda DESBALANCEADO: una de cada ocho filas es un éxito. Un
  modelo que siempre responda 'no tiene éxito' acertaría el 87,5 %, así que
  en el modelado hay que medir con recall, F1 y matriz de confusión, no solo
  con exactitud.""")

integrado = integrado.drop(columns=["_merge"])
integrado.to_csv(RUTA_SALIDA, index=False)
print(f"\n Guardado: {RUTA_SALIDA.relative_to(RAIZ)}  "
      f"({len(integrado)} filas x {integrado.shape[1]} columnas)")
print("\n Columnas del dataset integrado:")
for origen, cols in [("de vgsales_limpio", [c for c in integrado.columns
                                            if c in ventas.columns or c in
                                            ("name_clave", "genre_clave", "exito")]),
                     ("de ratings_limpio", [c for c in integrado.columns
                                            if c in COLUMNAS_RECEPCION])]:
    print(f"  {origen:18} ({len(cols):2}): {', '.join(cols)}")
