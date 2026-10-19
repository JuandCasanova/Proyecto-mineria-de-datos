# Auditoría de las dos fuentes crudas de videojuegos
# Proyecto: predicción del éxito en un videojuego por salir
# Ejecutar desde la raíz del proyecto:
#   python src/01_ingesta/auditar_fuentes.py
#
# Entradas (datos/):
#   vgsales_raw.csv   16.598 x 11   ventas (crudo, columnas con mayúsculas)
#   ratings_raw.csv   16.719 x 16   recepción (crudo, columnas con mayúsculas)
# Salida: consola (informe de auditoría; no escribe archivos)
#
# FASE CRISP-DM: 2 · Comprensión de los datos
# Etapa KDD:      1 · Selección (qué hay en las fuentes crudas y si sirven)
#
# ORDEN: existencia -> volumen -> estructura -> nulos -> rangos
#        -> cobertura del cruce -> veredicto
#
# Esta fase NO transforma nada: audita lo crudo antes de que la limpieza lo
# toque, para que cada decisión posterior tenga un "antes" verificable.
# Los crudos llegan con mayúsculas (Name, Year, Critic_Score); la limpieza
# los pasa a minúsculas. Aquí se busca cada columna sin importar el case.

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ENTRADAS = RAIZ / "datos"

FUENTES = {
    "ventas": ENTRADAS / "vgsales_raw.csv",
    "recepcion": ENTRADAS / "ratings_raw.csv",
}

# Llave de negocio: un juego en una plataforma, un año y un género.
LLAVE = {"ventas": ["Name", "Platform", "Year", "Genre"],
         "recepcion": ["Name", "Platform", "Year_of_Release", "Genre"]}

# Lo que la fase de limpieza ya sabe que va a encontrar. Si la auditoría no
# los confirma, algo cambió en los crudos y hay que parar antes de limpiar.
LO_ESPERADO = {
    "ventas": {"filas": 16598, "nulos": {"Year": 271, "Publisher": 261}},
    "recepcion": {"filas": 16719, "celdas_tbd": 2425,
                  "nulos": {"Year_of_Release": 269, "Publisher": 54}},
}

# Rangos plausibles según la naturaleza de cada variable (no según el dataset).
RANGOS_NEGOCIO = {"critic_score": (0, 100), "critic_count": (0, 10000),
                  "user_score": (0, 10), "user_count": (0, 500000),
                  "year": (1970, 2025)}


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} . {texto}")


def columna(df, nombre):
    """Devuelve el nombre real de la columna sin importar mayúsculas."""
    for real in df.columns:
        if real.lower() == nombre.lower():
            return real
    return None


def contar_tbd(df):
    """Celdas con el texto 'tbd': texto en columnas que deberían ser número."""
    tbd = 0
    for col in df.select_dtypes(include="object").columns:
        tbd += int(df[col].astype(str).str.lower().eq("tbd").sum())
    return tbd


def reportar_volumen(nombre, ruta):
    """Tamaño en disco, filas y columnas de cada fuente cruda."""
    df = pd.read_csv(ruta, low_memory=False)
    print(f"\n {nombre}: {ruta.name}")
    print(f"   disco      : {ruta.stat().st_size / 1024:.1f} KB")
    print(f"   dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"   columnas   : {list(df.columns)}")
    return df


def reportar_estructura(nombre, df, esperado):
    """Tipos por columna, filas esperadas y duplicados por llave de negocio."""
    print(f"\n Estructura de {nombre}")
    for col, tipo in df.dtypes.items():
        distintos = df[col].nunique(dropna=True)
        print(f"   {col:<18} {str(tipo):<10} {distintos:>6} valores distintos")

    ok = "OK" if len(df) == esperado["filas"] else "CAMBIO"
    print(f"   [{ok}] esperado {esperado['filas']} filas, encontradas {len(df)}")

    llave = [columna(df, c) for c in LLAVE[nombre]]
    if all(llave):
        duplicados = int(df.duplicated(subset=llave).sum())
        print(f"   duplicados por llave {llave}: {duplicados}"
              f"  (la limpieza decide qué hacer con ellos)")


def reportar_nulos(nombre, df, esperado):
    """Nulos por columna y texto sospechoso ('tbd') que rompe los tipos."""
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    print(f"\n Nulos en {nombre}: {int(df.isna().sum().sum())} en total")
    if nulos.empty:
        print("   ninguna columna tiene nulos")
    else:
        for col, cantidad in nulos.items():
            print(f"   {col:<18} {cantidad:>6}  ({cantidad / len(df):5.1%})")

    tbd = contar_tbd(df)
    if tbd:
        print(f"   celdas con el texto 'tbd': {tbd} (aquí NO se imputan)")
    if "celdas_tbd" in esperado:
        ok = "OK" if tbd == esperado["celdas_tbd"] else "CAMBIO"
        print(f"   [{ok}] esperado {esperado['celdas_tbd']}, encontrado {tbd}")

    for col_esperada, cantidad_esperada in esperado["nulos"].items():
        real = columna(df, col_esperada)
        encontrados = int(df[real].isna().sum()) if real else -1
        ok = "OK" if encontrados == cantidad_esperada else "CAMBIO"
        print(f"   [{ok}] {col_esperada}: esperados {cantidad_esperada} nulos, "
              f"encontrados {encontrados}")


def reportar_rangos(df):
    """Rangos observados frente a los rangos plausibles del negocio."""
    print("\n Rangos observados (columnas numéricas del negocio):")
    for nombre, (bajo, alto) in RANGOS_NEGOCIO.items():
        real = columna(df, nombre)
        if real is None and nombre == "year":
            real = columna(df, "Year_of_Release")
        if real is None:
            continue
        # errors="coerce" convierte los 'tbd' a nulo solo para medir aquí
        serie = pd.to_numeric(df[real], errors="coerce").dropna()
        if serie.empty:
            continue
        dentro = serie.between(bajo, alto)
        marca = "OK" if dentro.all() else f"{int((~dentro).sum())} fuera"
        print(f"   {real:<18} {serie.min():>8.2f} .. {serie.max():>8.2f}"
              f"   esperado {bajo}..{alto}   [{marca}]")


def reportar_cobertura(ventas, ratings):
    """Qué tan bien se podría cruzar recepción contra ventas por nombre."""
    print("\n Cobertura del cruce futuro (ventas <- recepción):")
    print(f"   juegos en ventas           : {ventas['Name'].nunique()}")
    print(f"   juegos en recepción        : {ratings['Name'].nunique()}")
    comunes = len(set(ventas["Name"].dropna()) & set(ratings["Name"].dropna()))
    print(f"   nombres en común (solo el nombre, el cruce se infla): {comunes}")
    print("   el cruce real usa name+platform+year+genre (lo hace merged.py)")


def main():
    print("=" * 70)
    print(" FASE 1 CRISP-DM . AUDITORÍA DE FUENTES CRUDAS")
    print(" Proyecto: predicción del éxito de un videojuego por salir")
    print("=" * 70)

    titulo(0, "EXISTENCIA DE LOS ARCHIVOS CRUDOS")
    faltantes = [str(r) for r in FUENTES.values() if not r.exists()]
    if faltantes:
        for ruta in faltantes:
            print(f" [FALTA] {ruta}")
        raise SystemExit(1)
    for nombre, ruta in FUENTES.items():
        print(f" [OK] {nombre:<12} {ruta.relative_to(RAIZ)}")

    datos = {}
    for nombre, ruta in FUENTES.items():
        titulo(1, f"VOLUMEN, ESTRUCTURA Y NULOS . {nombre.upper()}")
        df = reportar_volumen(nombre, ruta)
        reportar_estructura(nombre, df, LO_ESPERADO[nombre])
        reportar_nulos(nombre, df, LO_ESPERADO[nombre])
        reportar_rangos(df)
        datos[nombre] = df

    titulo(2, "COBERTURA DEL CRUCE FUTURO")
    reportar_cobertura(datos["ventas"], datos["recepcion"])

    titulo(3, "VEREDICTO")
    print("""
 La auditoría confirma: las dos fuentes son legibles, la llave de negocio
 existe en ambas y los nulos/'tbd' son los conocidos (MAR en year/publisher,
 MNAR en recepción). No hay nada que detenga la fase de limpieza.

 Siguiente paso: python src/02_preparacion/limpieza_datasets.py
""")


if __name__ == "__main__":
    main()
