import pandas as pd
from prophet import Prophet

df = pd.read_csv(r'C:\Users\guill\OneDrive\Escritorio\Cuarto\TFG (CTC)\Código\example_wp_log_peyton_manning.csv')
                 
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)

fig1.show()
input("Press Enter to CLOSE...")

fig2 = m.plot_components(forecast)
fig2.show()
input("Press Enter to CLOSE...")

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

plot_components_plotly(m, forecast).show()
input("Press Enter to CLOSE...")
