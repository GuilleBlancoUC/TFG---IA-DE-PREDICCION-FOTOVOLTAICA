# Data manipulation
# ==============================================================================
import pandas as pd
import numpy as np
from skforecast.datasets import fetch_dataset

# Plotting
# ==============================================================================
import matplotlib.pyplot as plt
import shap
from skforecast.plot import set_dark_theme

# Modeling and forecasting
# ==============================================================================
import sklearn
import lightgbm
import skforecast
from sklearn.inspection import PartialDependenceDisplay
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

color = '\033[1m\033[38;5;208m'
print(f"{color}Version skforecast: {skforecast.__version__}")
print(f"{color}Version scikit-learn: {sklearn.__version__}")
print(f"{color}Version lightgbm: {lightgbm.__version__}")
print(f"{color}Version shap: {shap.__version__}")

# ==============================================================================
#DATA
# Download data
# ==============================================================================
data = fetch_dataset(name="vic_electricity")
data.head(3)

#Frecuencia diaria
# Aggregation to daily frequency
data = data.resample('D').agg({'Demand': 'sum', 'Temperature': 'mean'})
data.head(3)

#Crear variable dia de la semana
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data.head(3)

#Separar en train y test
end_train = '2014-12-01 23:59:00'
data_train = data.loc[:end_train, :]
data_test = data.loc[end_train:, :]
print(f"Dates trains: {data_train.index.min()} - {data_train.index.max()}")
print(f"Dates test: {data_test.index.min()} - {data_test.index.max()}")

# ==============================================================================
# FORECASTING MODEL
window_features = RollingFeatures(stats=['mean'], window_sizes=24)
exog_features = ['Temperature', 'day_of_week', 'month']
forecaster = ForecasterRecursive(
                 regressor       = LGBMRegressor(random_state=123, verbose=-1),
                 lags            = 7,
                 window_features = window_features
             )

forecaster.fit(
    y    = data_train['Demand'],
    exog = data_train[exog_features],
)
forecaster

# ==============================================================================
#Model specific feature importances
importance = forecaster.get_feature_importances()
importance

# ==============================================================================
#SHAP feature importance in the overall model

X_train, y_train = forecaster.create_train_X_y(
                        y    = data_train['Demand'],
                        exog = data_train[exog_features],
                    )
print(X_train.head(3)) #Features
print(y_train.head(3)) #Target

#SHAP explainer
explainer = shap.Explainer(forecaster.regressor)

#50% of teh data
rng = np.random.default_rng(seed=785412)
sample= rng.choice(X_train.index, size=int(len(X_train)*0.5), replace=False)
X_train_sample = X_train.loc[sample, :]
shap_values = explainer.shap_values(X_train_sample)

# ==============================================================================
#SHAP summary plot
shap.initjs()
shap.summary_plot(shap_values, X_train_sample, max_display=10, show=False)
fig, ax = plt.gcf(), plt.gca()
ax.set_title('SHAP summary plot')
ax.tick_params(labelsize=8)
fig.set_size_inches(6,3)
#plt.show()

plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size=(6, 3))
#plt.show()

# ==============================================================================
#SHAP dependence plot
fig, ax = plt.subplots(figsize=(6, 3))

shap.dependence_plot("Temperature", shap_values, X_train_sample, ax=ax)
plt.show()

# ==============================================================================
#SHAP explanations for individual predictions
# Forecasting next 30 days
predictions = forecaster.predict(steps=30, exog=data_test[exog_features])
set_dark_theme()
fig, ax = plt.subplots(figsize=(6, 2.5))
data_test['Demand'].plot(ax=ax, label='Test')
predictions.plot(ax=ax, label='Predictions', linestyle='--')
ax.set_xlabel(None)
ax.legend();
plt.show()

#Input matrix to forecast the next 30 steps

X_predict = forecaster.create_predict_X(
                    steps = 30,
                    exog  = data_test[exog_features])
X_predict.head(3)

#Shap values for the predictions
shap_values = explainer.shap_values(X_predict)

#Waterfall plot for the first prediction
predicted_date = '2014-12-28'
iloc_predicted_date = X_predict.index.get_loc(predicted_date)
shap_values_single = explainer(X_predict)
shap.plots.waterfall(shap_values_single[iloc_predicted_date],show=False)
fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(6,3.5)
ax_list = fig.axes
ax = ax_list[0]
ax.tick_params(labelsize=10)
ax.set
plt.show()

#Forceplot for a single prediction
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values_single.values[iloc_predicted_date,],
    features=X_predict.iloc[iloc_predicted_date, :])

#Forceplot for tge 30 predictions
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values,
    features=X_predict)

# ==============================================================================
#SHAP values of backtesting_forecaster() output
cv = TimeSeriesFold(steps= 24, initial_train_size = len(data.loc[:'2014-12-01 23:59:00']))
_, predictions = backtesting_forecaster(
                        forecaster        = forecaster,
                        y                 = data['Demand'],
                        exog              = data[exog_features],
                        cv                = cv,
                        metric            = 'mean_absolute_error',
                        return_predictors = True,
                )
predictions.head(3)

#Waterfall for a single prediction generated during backtetsting
predictions = predictions.astype(data[exog_features].dtypes) # Ensure that the types are the same
iloc_predicted_date = predictions.index.get_loc('2014-12-16')
shap_values_single = explainer(predictions.iloc[:, 2:])
shap.plots.waterfall(shap_values_single[iloc_predicted_date], show=False)
fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(6, 3.5)
ax_list = fig.axes
ax = ax_list[0]
ax.tick_params(labelsize=8)
ax.set
plt.show()

# ==============================================================================
#Scikit-learn partial dependence plots
fig, ax = plt.subplots(figsize=(8, 3))
pd.plots = PartialDependenceDisplay.from_estimator(
    estimator = forecaster.regressor,
    X         = X_train,
    features  = ["Temperature", "lag_1"],
    kind      = 'both',
    ax        = ax,
)
ax.set_title("Partial Dependence Plot")
fig.tight_layout()
plt.show()

# ==============================================================================
#Session information
import session_info
session_info.show(html=False)