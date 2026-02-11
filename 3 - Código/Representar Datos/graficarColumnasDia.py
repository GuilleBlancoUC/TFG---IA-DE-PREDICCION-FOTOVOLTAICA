import pandas as pd
import matplotlib.pyplot as plt
import os

direccion = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(direccion,'..','CTC_processed.csv') #Lorca_processed para el de Lorca, Murcia
                              #CTC_processed.csv para el de Santander, Cantabria
data = pd.read_csv(dataset, sep=";", low_memory=False)

data["datetime"] = pd.to_datetime(data["datetime"], dayfirst=True, errors="coerce")

col_name = "solarRad" #Nombre de la columna

start_date = pd.to_datetime("11/05/2025", dayfirst=True)
end_date   = pd.to_datetime("14/05/2025", dayfirst=True)
mask = (data["datetime"] >= start_date) & (data["datetime"] <= end_date)
data_range = data.loc[mask]

plt.figure(figsize=(10, 4))
plt.plot(data_range["datetime"], data_range[col_name], marker=".", linestyle="-")
plt.xlabel("Fecha y hora")
plt.ylabel(col_name)
plt.title(f"Evolución de '{col_name}' entre {start_date.date()} y {end_date.date()}")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
