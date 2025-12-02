import pandas as pd


# Cargar datos de ejemplo (ajusta rutas o fuentes)
df_gen = pd.read_csv("Power_genCTC.csv", sep=";")       # columnas: timestamp, P, ...
df_met = pd.read_csv("MeteorologicoCTC.csv", sep=";")     # columnas: timestamp, temp, irr, hum, ...

# 1. Asegurar tipo datetime y ordenar por tiempo
for df in (df_gen, df_met):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

# 2. Merge por tiempo más cercano
# direction="nearest" busca antes o después
# tolerance limita la distancia máxima permitida (aquí 10 minutos)
result = pd.merge_asof(
    df_gen,
    df_met,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("3min"),
)

print(result.info())


cols = ["temperature", "windSpeed", "solarRad", "humedad", "rain", "pressure"]

for c in cols:
    # Pasar a string por si hay NaN
    s = df[c].astype(str)
    # 1) Quitar todo lo que no sea dígito, punto o coma
    s = s.str.replace(r"[^0-9\.,-]", "", regex=True)
    # 2) Si tu separador decimal es la coma, pásalo a punto
    s = s.str.replace(",", ".", regex=False)
    # 3) Convertir a float (no convertibles -> NaN)
    df[c] = pd.to_numeric(s, errors="coerce")

print(df[cols].dtypes)

result = result.drop(columns=cols, errors="ignore")
result = result.merge(
    df_met[["datetime"] + cols],   # columnas limpias
    on="datetime",
    how="left",
)

print(result.describe())

# 3. Guardar resultado si quieres
#result.to_csv("generacion_meteo_merged2min.csv", index=False)

