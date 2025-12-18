import pandas as pd
import matplotlib.pyplot as plt

dataset = 'Lorca_processed.csv' #Lorca_original.csv para el de Lorca, Murcia
                                #CTC_processed.csv para el de Santander, Cantabria
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
