import pandas as pd

df_gen = pd.read_csv("Power_genCTC.csv", sep=";")   # tiene columna Power_gen (o P)
df_met = pd.read_csv("MeteorologicoCTC.csv", sep=";")

# Asegurar datetime y ordenar
for df in (df_gen, df_met):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

# 1) Hacer merge_asof con METEOROLÓGICO como izquierda para no perder noches
result = pd.merge_asof(
    df_met,          # izquierda: todas las filas meteo
    df_gen,          # derecha: se une la potencia si hay timestamp cercano
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("3min")
)

# 2) Rellenar con 0 cuando no haya generación asociada (filas nocturnas)
result["Power_gen"] = result["Power_gen"].fillna(0)

print(result.info())


cols = ["temperature", "windSpeed", "solarRad", "humedad", "rain", "pressure"]

for c in cols:
    # Pasar a string por si hay NaN
    s = df[c].astype(str)
    # 1) Quitar todo lo que no sea dígito, punto o coma
    s = s.str.replace(r"[^0-9\.,-]", "", regex=True)
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
result.to_csv("generacion_meteo_merged2min.csv", index=False)

