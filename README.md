# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

data = sm.datasets.sunspots.load_pandas().data
data.index = pd.to_datetime(data['YEAR'], format='%Y')
data = data[['SUNACTIVITY']] 

result = adfuller(data['SUNACTIVITY'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['SUNACTIVITY'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['SUNACTIVITY'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['SUNACTIVITY'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
mse = mean_squared_error(test_data['SUNACTIVITY'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data['SUNACTIVITY'], label='Test Data - Sunspots')
plt.plot(predictions, label='Predictions - Sunspots', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Number of Sunspots')
plt.title('AR Model Predictions vs Test Data - Sunspots')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
## Autocorrelation Function (ACF)
<img width="693" height="534" alt="image" src="https://github.com/user-attachments/assets/79bece2e-4355-44de-861c-4745f7c3fa5d" />

## Partial Autocorrelation Function (PACF)
<img width="725" height="527" alt="image" src="https://github.com/user-attachments/assets/ac3d3aad-5924-4400-b6b9-dc494409ec8a" />

## Prediction
<img width="750" height="388" alt="image" src="https://github.com/user-attachments/assets/2846fce3-cc2e-4c2b-a4c1-3762f9ea4b7e" />

## Performance Measure (MSE)
<img width="794" height="113" alt="image" src="https://github.com/user-attachments/assets/0e40701c-c31d-4709-9beb-f5ff81c3b1e7" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
