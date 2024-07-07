import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'F:/VSC/datasets1.csv'
df = pd.read_csv(file_path)

# Aggregate the Amount by Date
#df_aggregated = df.groupby('Date', as_index=False)['Amount'].sum()

# Rename columns to fit Prophet's requirements
df.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)

# Ensure the date column is in datetime format
df['ds'] = pd.to_datetime(df['ds'])

print(df.describe())
# Initialize the Prophet model
model = Prophet()

# Fit the model on the dataset
model.fit(df)

# Create a dataframe with future dates for forecasting
future = model.make_future_dataframe(periods=30)  # Forecasting for the next 30 days

# Make the forecast
forecast = model.predict(future)


# Display the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
plt.show()