import pandas as pd

df_gen = pd.read_csv("Power_genCTC.csv", sep=";")
df_met = pd.read_csv("MeteorologicoCTC.csv", sep=";")

for df in (df_gen, df_met):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

result = pd.merge_asof(
    df_met,          
    df_gen,          
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("3min")
)

result["Power_gen"] = result["Power_gen"].fillna(0)

print(result.info())

cols = ["temperature", "windSpeed", "solarRad", "humedad", "rain", "pressure"]

for c in cols:
    s = df[c].astype(str)
    s = s.str.replace(r"[^0-9\.,-]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    df[c] = pd.to_numeric(s, errors="coerce")

print(df[cols].dtypes)

result = result.drop(columns=cols, errors="ignore")
result = result.merge(
    df_met[["datetime"] + cols],
    on="datetime",
    how="left",
)

print(result.describe())

result.to_csv("generacion_meteo_merged2min.csv", index=False)