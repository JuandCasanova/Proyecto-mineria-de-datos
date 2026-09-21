import pandas as pd
from pathlib import Path

# 1. Cargar los datos desde el CSV
ruta = Path("Cleaned Data 1.csv")
df = pd.read_csv(ruta)

# 2. Dimensiones: filas x columnas
print("Dimensiones:", df.shape)

# 3. Primeras filas
print(df.head())

# 4. Tipos de datos y memoria
df.info()

# Estadística descriptiva de columnas numéricas
print(df.describe().round(1))

# Variable objetivo
print(df["Name"].value_counts(normalize=True).round(3))

# Cruce rápido:tasa de fuga por plataforma
print(pd.crosstab(df["Genre"], df["Name"], normalize="index").round(2))