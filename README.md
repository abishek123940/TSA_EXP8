# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 04.11.2025 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
data = pd.read_csv('World_Population.csv')

print("Columns in dataset:", list(data.columns))
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

data.columns = ['No', 'Country', 'Population', 'Yearly_Change', 'Net_Change',
                'Density', 'Land_Area', 'Migrants', 'Fert_Rate',
                'Med_Age', 'Urban_Pop', 'World_Share']

def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '').strip()
    try:
        return float(x)
    except:
        return np.nan

for col in ['Population', 'Yearly_Change', 'Net_Change', 'Density',
            'Land_Area', 'Migrants', 'Fert_Rate', 'Med_Age', 'World_Share']:
    data[col] = data[col].apply(clean_numeric)

data = data.dropna(subset=['Population'])

top_countries = data.nlargest(30, 'Population')

print("\nTop 5 Countries by Population:")
print(top_countries[['Country', 'Population', 'World_Share']].head())

plt.figure(figsize=(14, 7))
plt.barh(top_countries['Country'], top_countries['Population'], color='skyblue')
plt.xlabel('Population')
plt.ylabel('Country')
plt.title('Top 30 Countries by Population')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

top_countries_sorted = top_countries.sort_values(by='Population', ascending=False).reset_index(drop=True)
rolling_mean_3 = top_countries_sorted['Population'].rolling(window=3).mean()
rolling_mean_5 = top_countries_sorted['Population'].rolling(window=5).mean()

plt.figure(figsize=(12, 6))
plt.plot(top_countries_sorted.index + 1, top_countries_sorted['Population'], label='Original', marker='o')
plt.plot(top_countries_sorted.index + 1, rolling_mean_3, label='Moving Avg (3)', linestyle='--')
plt.plot(top_countries_sorted.index + 1, rolling_mean_5, label='Moving Avg (5)', linestyle=':')
plt.title('Moving Average of Population (Top 30 Countries)')
plt.xlabel('Rank (1 = Most Populous)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

scaler = MinMaxScaler()
scaled_pop = scaler.fit_transform(top_countries_sorted[['Population']]).flatten()

model = ExponentialSmoothing(scaled_pop, trend='add', seasonal=None)
fit_model = model.fit()
forecast = fit_model.forecast(5)

plt.figure(figsize=(12, 6))
plt.plot(scaled_pop, label='Scaled Population', marker='o')
plt.plot(range(len(scaled_pop), len(scaled_pop) + len(forecast)), forecast, label='Forecast (next 5 ranks)', color='red', marker='x')
plt.title('Exponential Smoothing on Population (Scaled)')
plt.xlabel('Country Rank Order')
plt.ylabel('Scaled Population')
plt.legend()
plt.grid()
plt.show()

print("\nDescriptive Statistics:")
print(top_countries_sorted['Population'].describe())

print("\nMean Population (Top 30):", top_countries_sorted['Population'].mean())
print("Standard Deviation:", top_countries_sorted['Population'].std())
```

### OUTPUT:
# First 5 Rows of dataset

<img width="1599" height="222" alt="image" src="https://github.com/user-attachments/assets/fb6516c7-ac1e-4de3-994a-b61fd0283e40" />

Moving Average :

<img width="1001" height="545" alt="download" src="https://github.com/user-attachments/assets/6021c629-3f69-477b-a26c-4357ed5c0900" />

Plot Transform Dataset :

<img width="1245" height="622" alt="download" src="https://github.com/user-attachments/assets/18cc57c0-0f6d-4270-a3ee-67b08b2d4ae8" />

Exponential Smoothing :

<img width="1001" height="545" alt="download" src="https://github.com/user-attachments/assets/18f7466d-7924-4e34-9c4c-0a8f542a91b9" />


### RESULT:

Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
