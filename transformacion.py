from pathlib import Path
import warnings

import pandas as pd
from sklearn.descomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RUTA=Path(__file__).parent.parent / "data" / "raw" / "vgsales_merged.csv"
NUMERICAS=[
    "NA_Sales"]

df=pd.read_csv(RUTA)
X=df[NUMERICAS]
print("Dataset limpio:", df.shape, "| matriz numérica:", X.shape)

#1. Escalamiento
def distancia(a, b):
    return ((a - b) ** 2).sum() ** 0.5
c1,c2= X.iloc[0], X.iloc[1]
print("Distancia entre c1 y c2:", distancia(c1, c2))