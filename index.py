import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv('tsla_2014_2023.csv')
# print(df.head())
print(df.shape)
print(df.info())

#date form dtype. obj -> datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print(df.head())

# Tesla stock price information from the year 2014 to 2023
plt.figure(figsize=(20, 15))
plt.plot(df.index, df['open'], label='Open')
plt.plot(df.index, df['close'], label='Close')
plt.plot(df.index, df['high'], label='High')
plt.plot(df.index, df['low'], label='Low')
plt.legend()
plt.show()


#correlation heatmap between features and label(next_day_close)
df_corr = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(df_corr, annot=True, fmt='.1f')
plt.show()


#splitting the dataset into features and labels.
y = df['next_day_close']
X = df.drop(columns='next_day_close')

#tarin test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)


# standardize our dataset using something called StandardScaler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print(X_train_scaled.shape, X_test_scaled.shape)


#model
#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)

actual = y_test
predicted = lr_prediction

# Plot the actual values as a scatter plot
plt.scatter(range(len(actual)), actual, color='blue', label='Actual')

# Plot the predicted values as a line
plt.scatter(range(len(actual)), predicted, color='red', label='Predicted')

# A line between the actual point and predicted point
for i in range(len(actual)):
    plt.plot([i, i], [actual.iloc[i], predicted[i]], color='green', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (Stock price prediction)')
plt.legend()
plt.show()


#metrics
lr_mae = mean_absolute_error(y_test, lr_prediction)
print(lr_mae) # 2.3867493377661844




# #Random Forest Regressor
# rfr = RandomForestRegressor()
# rfr.fit(X_train, y_train)
# rfr_prediction = rfr.predict(X_test)
# rfr_mae = mean_absolute_error(y_test, rfr_prediction)
# print(rfr_mae) #2.7267115297023783