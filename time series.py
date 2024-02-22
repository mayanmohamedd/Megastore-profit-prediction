from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa import stattools
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
import matplotlib as mpl

# # ----------------------------------------- Time Series ---------------------------------------------

t_series_data = pd.read_csv("megastore-regression-dataset.csv", parse_dates=['Order Date'],
                            index_col='Order Date')
t_series_data = t_series_data.drop(["Ship Date", "Country", "Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment",
                                    "City", "State", "Region", "Product ID", "CategoryTree", "Product Name", "Row ID", "Postal Code", "Sales", "Quantity", "Discount"], axis=1)

# plotting the data


def plot_t_series_data(t_series_data, x, y, title="", xlabel='Date', ylabel='Profit', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


plot_t_series_data(t_series_data, x=t_series_data.index,
                   y=t_series_data['Profit'], title='Profit from 2014 to 2018.')

sns.lineplot(t_series_data)
plt.ylabel("Profit")
plt.show()
t_series_data.reset_index(inplace=True)

# Plotting seasonal graph
# Prepare data
t_series_data['year'] = [d.year for d in t_series_data['Order Date']]
t_series_data['month'] = [d.strftime('%b')
                          for d in t_series_data['Order Date']]
years = t_series_data['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(
    list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16, 12), dpi=100)
for i, y in enumerate(years):
    if i > 0:
        plt.plot('month', 'Profit',
                 data=t_series_data.loc[t_series_data.year == y, :], color=mycolors[i], label=y)
        plt.text(t_series_data.loc[t_series_data.year == y, :].shape[0]-.9,
                 t_series_data.loc[t_series_data.year == y, 'Profit'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(-1000, 2000),
              ylabel='$Profit$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Profit Time Series", fontsize=20)
plt.show()
# -------------------------------------------------------------------------------------------

# Plotting year wise and month wise box plot
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(45, 30), dpi=100)
sns.boxplot(x='year', y='Profit', data=t_series_data, ax=axes[0])
sns.boxplot(x='month', y='Profit',
            data=t_series_data.loc[~t_series_data.year.isin([2014, 2018]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
# ----------------------------------------------------------------------------------------------
# Decomposing
decompose = seasonal_decompose(
    t_series_data['Profit'], model='additive', period=200)
decompose.plot()
plt.show()

# -----------------------------------------------------------------------------------------------
# Testing stationary
# At_series_data Test
result = adfuller(t_series_data['Profit'], autolag='AIC')
print(f'At_series_data Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'{key}, {value}')
# ----------------------------------------------------------------------------------------------------

# Auto-correlation
autocorrelation_lag1 = t_series_data['Profit'].autocorr(lag=1)
print("One Month Lag: ", autocorrelation_lag1)

autocorrelation_lag3 = t_series_data['Profit'].autocorr(lag=3)
print("Three Month Lag: ", autocorrelation_lag3)

autocorrelation_lag6 = t_series_data['Profit'].autocorr(lag=6)
print("Six Month Lag: ", autocorrelation_lag6)

autocorrelation_lag9 = t_series_data['Profit'].autocorr(lag=9)
print("Nine Month Lag: ", autocorrelation_lag9)

# plt.figure(figsize=(20, 20), dpi=100)
plot_acf(t_series_data['Profit'].tolist(), lags=50)
plt.show()
# -----------------------------------------------------------------------------------------------------------------

# Forecasting

# Fit the auto_arima model
model = auto_arima(t_series_data['Profit'], seasonal=True)

# Forecast future values
# Replace 12 with the desired number of periods to forecast
forecast_values = model.predict(n_periods=12)

# Print the forecasted values
print(forecast_values)

# Actual values
# Replace 12 with the number of periods forecasted
actual_values = t_series_data['Profit'][-12:]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))

# Print RMSE
print("RMSE:", rmse)
