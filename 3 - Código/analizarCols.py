import pandas as pd
import matplotlib.pyplot as plt

# data contiene al menos las columnas: 'hour' y la columna que quieres analizar
# col_name es el nombre de la columna que quieres ver, por ejemplo "Power_gen"
dataset = 'CTC_processed.csv' #Lorca_original.csv para el de Lorca, Murcia
                               #CTC.csv para el de Santander, Cantabria
data = pd.read_csv(
        dataset,
        sep=";",
        low_memory=False,
)

if "datetime" not in data.columns:
    data["Time"] = pd.to_datetime(data["Time"], dayfirst=True, errors="coerce") #convertimos time a datatime
    data["month"] = data["Time"].dt.month
    data["day"] = data["Time"].dt.day
    data["hour"] = data["Time"].dt.hour
    data["minute"] = data["Time"].dt.minute
else:
    data["datetime"] = pd.to_datetime(data["datetime"], dayfirst=True, errors="coerce") #convertimos time a datatime
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute

col_name = "Power_gen" 

# Calcular la media por hora del día
hourly_mean = data.groupby("hour")[col_name].mean().reset_index()

# Ordenar por hora por si acaso
hourly_mean = hourly_mean.sort_values("hour")

# Gráfico
plt.figure(figsize=(8, 4))
plt.plot(hourly_mean["hour"], hourly_mean[col_name], marker="o")
plt.xticks(range(0, 24))
plt.xlabel("Hora del día")
plt.ylabel(f"Media de {col_name}")
plt.title(f"Evolución media diaria de '{col_name}' por hora")
plt.grid(True)
plt.show()
