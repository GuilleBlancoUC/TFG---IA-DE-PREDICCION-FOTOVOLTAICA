import pandas as pd
import matplotlib.pyplot as plt

dataset = 'CTC_processed.csv'
data = pd.read_csv(dataset, sep=";", low_memory=False)

data["datetime"] = pd.to_datetime(data["datetime"], dayfirst=True, errors="coerce")

col_name = "solarRad"  # columna a analizar

# --- RANGO DE DÍAS A USAR ---
start_date = pd.to_datetime("12/05/2025", dayfirst=True)
end_date   = pd.to_datetime("14/05/2025", dayfirst=True)
mask = (data["datetime"] >= start_date) & (data["datetime"] <= end_date)
data_range = data.loc[mask]
# ----------------------------

plt.figure(figsize=(10, 4))
plt.plot(data_range["datetime"], data_range[col_name], marker=".", linestyle="-")
plt.xlabel("Fecha y hora")
plt.ylabel(col_name)
plt.title(f"Evolución de '{col_name}' entre {start_date.date()} y {end_date.date()}")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
