import pandas as pd
import matplotlib.pyplot as plt
import os

direccion = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(direccion,'..','PlantaA_processed.csv') #Lorca_processed.csv para el de Lorca, Murcia
                                #PlantaA_processed.csv para el emplazamiento A anonimizado
data = pd.read_csv(
        dataset,
        sep=";",
        low_memory=False,
)


data["datetime"] = pd.to_datetime(data["datetime"], dayfirst=True, errors="coerce") 
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day
data["hour"] = data["datetime"].dt.hour
data["minute"] = data["datetime"].dt.minute

col_name = "solarRad" #columna a analizar

hourly_mean = data.groupby("hour")[col_name].mean().reset_index()

hourly_mean = hourly_mean.sort_values("hour")

plt.figure(figsize=(8, 4))
plt.plot(hourly_mean["hour"], hourly_mean[col_name], marker="o")
plt.xticks(range(0, 24))
plt.xlabel("Hora del día")
plt.ylabel(f"Media de {col_name}")
plt.title(f"Evolución media diaria de '{col_name}' por hora")
plt.grid(True)
plt.show()
