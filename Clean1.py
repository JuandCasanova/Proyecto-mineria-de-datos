import pandas as pd

INPUT_FILE = 'Cleaned Data 1.csv'
OUTPUT_FILE = 'Cleaned Data Final.csv'

def clean_data(input_file):
    df = pd.read_csv(input_file, na_values=['', ' ', 'N/A', 'NA', 'null', 'tbd'])

    #Usa nombres de columna predecibles a lo largo del proceso de limpieza.
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.strip('_')
    )

    #Remueve filas y columnas sin información.
    df = df.dropna(how='all').dropna(axis=1, how='all')

    string_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        df[col] = df[col].astype('string').str.strip()

    numeric_cols = [
        'year_of_release', 'na_sales', 'eu_sales', 'jp_sales',
        'other_sales', 'global_sales', 'critic_score', 'critic_count',
        'user_score', 'user_count'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype('string').str.replace(',', '', regex=False),
                errors='coerce'
            )
    df = df.drop_duplicates().reset_index(drop=True)
    return df


df = clean_data(INPUT_FILE)
df.to_csv(OUTPUT_FILE, index=False)

print(f'Datos limpios: {df.shape[0]} filas y {df.shape[1]} columnas')
print(f'Archivo generado: {OUTPUT_FILE}')
print('\nValores faltantes por columna:')
print(df.isna().sum())

df['year_of_release'] = pd.to_numeric(
    df['year_of_release'],
    errors='coerce'
)
# Conversión de fechas
df['year_of_release'] = pd.to_datetime(df['year_of_release'], errors='coerce', format='%Y-%m-%d')
