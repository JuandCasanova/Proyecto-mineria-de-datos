import pandas as pd

INPUT_FILE = 'vgsales.csv'
OUTPUT_FILE = 'vgsales_cleaned.csv'

def clean_data(input_file):
	df = pd.read_csv(
		input_file,
		na_values=['', ' ', 'N/A', 'NA', 'null', 'Unknown']
	)
	# Usa nombres de columna predecibles a lo largo del proceso de limpieza.
	df.columns = (
		df.columns.str.strip()
		.str.lower()
		.str.replace(r'[^a-z0-9]+', '_', regex=True)
		.str.strip('_')
	)

	df = df.dropna(how='all').dropna(axis=1, how='all')

	string_cols = df.select_dtypes(include=['object', 'string']).columns
	for col in string_cols:
		df[col] = df[col].astype('string').str.strip()

	numeric_cols = [
		'rank', 'year', 'na_sales', 'eu_sales', 'jp_sales',
		'other_sales', 'global_sales'
	]
	for col in numeric_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')

	#Solo las filas duplicadas exactas son eliminadas.
	df = df.drop_duplicates().reset_index(drop=True)
	return df
df = clean_data(INPUT_FILE)
df.to_csv(OUTPUT_FILE, index=False)

print(f'Datos limpios: {df.shape[0]} filas y {df.shape[1]} columnas')
print(f'Archivo generado: {OUTPUT_FILE}')
print('\nValores faltantes por columna:')
print(df.isna().sum())
