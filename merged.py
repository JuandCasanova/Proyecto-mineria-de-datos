import pandas as pd


SALES_FILE = 'vgsales_cleaned.csv'
RATINGS_FILE = 'Cleaned Data Final.csv'
OUTPUT_FILE = 'vgsales_merged.csv'


sales = pd.read_csv(SALES_FILE)
ratings = pd.read_csv(RATINGS_FILE)

# Unify the year column name and types before matching the datasets.
ratings = ratings.rename(columns={'year_of_release': 'year'})
for data in (sales, ratings):
	data['name'] = data['name'].astype('string').str.strip()
	data['genre'] = data['genre'].astype('string').str.strip()
	data['year'] = pd.to_numeric(data['year'], errors='coerce')

merge_keys = ['name', 'year', 'genre']
rating_columns = merge_keys + [
	'critic_score',
	'critic_count',
	'user_score',
	'user_count',
	'developer',
	'rating'
]

# Keep only rating information from the second dataset. Sales already exists
# in vgsales_cleaned.csv, so this avoids duplicate columns in the result.
ratings = ratings[rating_columns].drop_duplicates(subset=merge_keys)

# Keep every sales record, including records without a matching rating.
merged = sales.merge(
	ratings,
	on=merge_keys,
	how='left',
	indicator=True,
	validate='many_to_one'
)

matches = (merged['_merge'] == 'both').sum()
without_match = (merged['_merge'] == 'left_only').sum()
merged = merged.drop(columns='_merge')
merged.to_csv(OUTPUT_FILE, index=False)

print(f'Archivo generado: {OUTPUT_FILE}')
print(f'Filas combinadas: {len(merged)}')
print(f'Coincidencias con calificaciones: {matches}')
print(f'Filas sin coincidencia: {without_match}')
