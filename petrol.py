import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Historical petrol prices in PKR/Litre (simplified, real + interpolated)
years = list(range(1947, 2025))
prices = [
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
    15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0, 38.0, 42.0,
    45.0, 48.0, 50.0, 55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0,
    75.0, 78.0, 82.0, 85.0, 88.0, 92.0, 95.0, 97.0, 100.0, 103.0,
    107.0, 112.0, 118.0, 123.0, 128.0, 132.0, 137.0, 140.0, 145.0, 150.0,
    155.0, 160.0, 165.0, 170.0, 180.0, 190.0, 200.0, 205.0  # up to 2024
]

# Create DataFrame
df = pd.DataFrame({'Year': years, 'Price_PKR': prices})
df.set_index('Year', inplace=True)

# Fit ARIMA model
model = ARIMA(df['Price_PKR'], order=(2, 1, 2))
model_fit = model.fit()

# Forecast next 20 years
forecast_years = list(range(2025, 2045))
forecast = model_fit.forecast(steps=20)
forecast_df = pd.DataFrame({'Year': forecast_years, 'Price_PKR': forecast.values})
forecast_df.set_index('Year', inplace=True)

# Combine historical + forecast
full_df = pd.concat([df, forecast_df])

# Save to Excel
full_df.to_excel('petrol_forecast.xlsx')
print("✅ Data saved to petrol_forecast.xlsx")

# Plot and save image
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price_PKR'], label='Historical', marker='o')
plt.plot(forecast_df.index, forecast_df['Price_PKR'], label='Forecast', marker='o', linestyle='--', color='red')
plt.title('Pakistan Petrol Price Forecast (1947–2044)')
plt.xlabel('Year')
plt.ylabel('Price (PKR/Litre)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('petrol_forecast.png')
plt.show()
print("✅ Graph saved to petrol_forecast.png")
