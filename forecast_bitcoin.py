
#%% Import modules

from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

def grangers_causality_matrix(data, variables, maxlag, test = 'ssr_chi2test', verbose = False):

    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns = variables, index = variables)

    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag = maxlag, verbose = False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]

    return dataset


#%% Read data and rename columns according to NeuralProphet

df = pd.read_csv('data/all_data.csv', sep = ';', usecols = [0,1,2,3,4,8])
df.rename(columns = {'date': 'ds', 'Close': 'y'}, inplace = True)


#%% Normalize data

norm_df = df[df.columns[1:]]
min_vals = norm_df.min()
max_vals = norm_df.max()
norm_df = (norm_df - min_vals)/(max_vals - min_vals)
norm_df['ds'] = pd.to_datetime(df['ds'], format = "%d/%m/%Y")
norm_df = norm_df[norm_df.columns[::-1]]


#%% Plot the data all together for a first quick look

norm_df.plot(x = 'ds', y = [2,3,4,5], alpha = 0.4)
plt.plot(norm_df['ds'], norm_df['y'])
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Normalized value", fontsize = 18)
plt.title("Platform's traffics vs Bitcoin price", fontsize = 20)
plt.legend(['Twitter', 'Wikipedia', 'Google', 'Reddit', 'y'], fontsize = 18)
plt.show()


#%% Calculate correlation between columns and plot heatmap

plt.figure()
c = norm_df.corr()
ax = plt.axes()
ax.set_title("Cross-correlation", fontsize = 20)
sns.heatmap(c['y'].values.reshape(-1,1), yticklabels = c.columns, ax = ax, annot = True)
plt.show()


#%% Perform Granger's causality test

print("Granger's causality matrix:")
print(grangers_causality_matrix(norm_df, variables = norm_df.columns[1:], maxlag = 14))


#%% Drop Wikipedia and Reddit data for this case

norm_df = norm_df.drop(columns = ['Reddit', 'Wikipedia'])


#%% Step 1: Overview of training fit

# Initialize NeuralProphet model and add platform traffics as lagged regressors
m = NeuralProphet(n_forecasts = 14, n_lags = 14, daily_seasonality = False, seasonality_mode = "multiplicative")
    
m = m.add_lagged_regressor(name = 'Google')
m = m.add_lagged_regressor(name = 'Twitter')

# Fit model and predict
metrics = m.fit(norm_df, freq = "D")
future = m.make_future_dataframe(norm_df, periods = 14, n_historic_predictions = True)
forecast = m.predict(future)

# Plot fitted against actual data
plt.figure()
ax = plt.axes()
m.plot(forecast, ax = ax, xlabel = "Date", ylabel = "Price (normalized)")
ax.set_title("Regression fit to training data")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(18)

# Plot fitted against actual data in a regression line 
train_act = df['y'][:2541]
train_fit = forecast['yhat14'][:2541].fillna(0)
train_fit = (max_vals['y'] - min_vals['y']) * np.array(train_fit) + min_vals['y']

plt.figure()
plt.grid()
plt.scatter(train_act, train_fit)
plt.plot(train_act, train_act, color = "red")
plt.xlabel("Actual values", fontsize = 18)
plt.ylabel("Fitted values", fontsize = 18)
plt.title("Regression fit to training data", fontsize = 20)

#%% Step 2: Train simple model and make forecasts based on 2-weeks windows

fcst_1 = []
test_1 = norm_df[norm_df['ds'] >= pd.to_datetime('15/9/2020', format = "%d/%m/%Y")]
test_1 = test_1[test_1['ds'] < pd.to_datetime('15/12/2020', format = "%d/%m/%Y")]

for d in test_1['ds'][::14]:
    
    print("Predicting week", d.date())
    print("--------------------------")
    
    # Keep all data up until current date as training data 
    train_1 = norm_df[norm_df['ds'] < d]
    train_1 = train_1.drop(columns = ['Google', 'Twitter'])
    
    # Initialize NeuralProphet model and add platform traffics as lagged regressors
    m = NeuralProphet(n_forecasts = 14, n_lags = 14, daily_seasonality = False, seasonality_mode = "multiplicative")
    
    # Fit model and predict
    metrics = m.fit(train_1, freq = "D")
    future = m.make_future_dataframe(train_1, periods = 14)
    forecast = m.predict(future)
    
    # Append weekly forecast to the list of forecasts
    for i in range(14):
        fcst_1.append(forecast['yhat' + str(i+1)][i + 14])

fcst_1 = fcst_1[:len(test_1)]

# Denormalize results for better presentation
fcst_1 = (max_vals['y'] - min_vals['y']) * np.array(fcst_1) + min_vals['y']
test_1['y'] = (max_vals['y'] - min_vals['y']) * test_1['y'] + min_vals['y']

# Calculate MAP error
mape_1 = np.mean(np.abs((test_1['y'] - fcst_1) / test_1['y'])) * 100

# Plot 14-days forecasts against actual data
fig, ax = plt.subplots(figsize = (20,10))
plt.plot(test_1['ds'], fcst_1, test_1['ds'], test_1['y'], linewidth = 4)
plt.grid()
plt.title("Forecasts of simple model", fontsize = 24)
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Price ($)", fontsize = 18)
plt.legend(["14-days forecasts", "Actual price"], loc = 'upper left', fontsize = 18)
ax.text(0.66, 0.05, str("Mean absolute percentage error: %.2f" % mape_1) + "%", transform = ax.transAxes, fontsize = 18,
        va = 'top', bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))
plt.show()


#%% Step 3: Add regressors and repeat the above steps 

fcst_2 = []
test_2 = norm_df[norm_df['ds'] >= pd.to_datetime('15/9/2020', format = "%d/%m/%Y")]
test_2 = test_2[test_2['ds'] < pd.to_datetime('15/12/2020', format = "%d/%m/%Y")]

for d in test_2['ds'][::14]:
    
    print("Predicting week", d.date())
    print("--------------------------")
    
    # Keep all data up until current date as training data 
    train_2 = norm_df[norm_df['ds'] < d]
    
    # Initialize NeuralProphet model and add platform traffics as lagged regressors
    m = NeuralProphet(n_forecasts = 14, n_lags = 14, daily_seasonality = False, seasonality_mode = "multiplicative")
    
    m = m.add_lagged_regressor(name = 'Google')
    m = m.add_lagged_regressor(name = 'Twitter')
    
    # Fit model and predict
    metrics = m.fit(train_2, freq = "D")
    future = m.make_future_dataframe(train_2, periods = 14)
    forecast = m.predict(future)
    
    # Append weekly forecast to the list of forecasts
    for i in range(14):
        fcst_2.append(forecast['yhat' + str(i+1)][i + 14])

fcst_2 = fcst_2[:len(test_2)]

# Denormalize results for better presentation
fcst_2 = (max_vals['y'] - min_vals['y']) * np.array(fcst_2) + min_vals['y']
test_2['y'] = (max_vals['y'] - min_vals['y']) * test_2['y'] + min_vals['y']

# Calculate MAP error
mape_2 = np.mean(np.abs((test_2['y'] - fcst_2) / test_2['y'])) * 100

# Plot 14-days forecasts against actual data
fig, ax = plt.subplots(figsize = (20,10))
plt.plot(test_2['ds'], fcst_2, test_2['ds'], test_2['y'], linewidth = 4)
plt.grid()
plt.title("Forecasts of model with external regressors", fontsize = 24)
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Price ($)", fontsize = 18)
plt.legend(["14-days forecasts", "Actual price"], loc = 'upper left', fontsize = 18)
ax.text(0.66, 0.05, str("Mean absolute percentage error: %.2f" % mape_2) + "%", transform = ax.transAxes, fontsize = 18,
        va = 'top', bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))
plt.show()
