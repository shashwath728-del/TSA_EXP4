# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 16-05-2026

### AIM:
To implement ARMA model in python.
### SOFTWARE REQUIRED:
google colab
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
NAME: Thejashree S
REGNO: 212224240175
```
```py
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/Walmart_Sales.csv', parse_dates=['Date'])

data=data.head(200)
X = data['Weekly_Sales']

plt.figure(figsize=(12, 6))
plt.plot(X)
plt.title('Original Data')
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_acf(X, lags=(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()


# Step 5: Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Step 6: Simulate ARMA(1,1)
N = 1000

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 4))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# ACF & PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.title("ACF - ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PACF - ARMA(1,1)")
plt.show()

# Step 7: Fit ARMA(2,2)

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Parameters:")
print(arma22_model.params)

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Step 8: Simulate ARMA(2,2)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.figure(figsize=(12, 4))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# ACF & PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.title("ACF - ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PACF - ARMA(2,2)")
plt.show()

```
### OUTPUT:
ORIGINAL DATA:

<img width="1296" height="658" alt="image" src="https://github.com/user-attachments/assets/6a5cdc92-ce0e-4399-b7eb-d40f0620d223" />
<img width="1444" height="678" alt="image" src="https://github.com/user-attachments/assets/a0735ecd-89d4-458f-85c2-d7fab3d7f37e" />

SIMULATED ARMA(1,1) PROCESS:

<img width="1234" height="417" alt="image" src="https://github.com/user-attachments/assets/abaf95bd-2a60-403c-b89f-026018a437a9" />


Autocorrelation

<img width="1196" height="508" alt="image" src="https://github.com/user-attachments/assets/2433d09b-1577-47b4-bfe0-2d0bfce10b3c" />

Partial Autocorrelation

<img width="834" height="496" alt="image" src="https://github.com/user-attachments/assets/3c79991a-fd75-4e51-93fa-c87fed5d3473" />


SIMULATED ARMA(2,2) PROCESS:

<img width="1286" height="429" alt="image" src="https://github.com/user-attachments/assets/3ac504bb-0eb3-4b0c-984c-6e07f7d9bfb4" />


Autocorrelation

<img width="1155" height="496" alt="image" src="https://github.com/user-attachments/assets/7f08695d-b932-49fc-99ec-4f63dfab8ae3" />


Partial Autocorrelation

<img width="1152" height="497" alt="image" src="https://github.com/user-attachments/assets/0239da3c-e8ed-4896-a501-bb3841b432a0" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
