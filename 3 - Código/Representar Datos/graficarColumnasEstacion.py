import pandas as pd
import matplotlib.pyplot as plt
import os

direccion = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(direccion,'..','CTC_processed.csv') #Lorca_processed para el de Lorca, Murcia
                                                            #CTC_processed.csv para el de Santander, Cantabria
data = pd.read_csv(dataset, sep=";", low_memory=False)

data["datetime"] = pd.to_datetime(data["datetime"], dayfirst=True, errors="coerce")
data["month"] = data["datetime"].dt.month
data["hour"] = data["datetime"].dt.hour

def get_season(month):
    if month in [12, 1, 2]:
        return "invierno"
    elif month in [3, 4, 5]:
        return "primavera"
    elif month in [6, 7, 8]:
        return "verano"
    else:
        return "otoño"

data["season"] = data["month"].apply(get_season)

col_name = "solarRad" #Nombre de la columna
season_to_plot = "otoño" #Estacion

data_season = data[data["season"] == season_to_plot]

hourly_mean = data_season.groupby("hour")[col_name].mean().reset_index()
hourly_mean = hourly_mean.sort_values("hour")

plt.figure(figsize=(8, 4))
plt.plot(hourly_mean["hour"], hourly_mean[col_name], marker="o")
plt.xticks(range(0, 24))
plt.xlabel("Hora del día")
plt.ylabel(f"Media de {col_name}")
plt.title(f"Evolución media diaria de '{col_name}' por hora en {season_to_plot}")
plt.grid(True)
plt.show()
