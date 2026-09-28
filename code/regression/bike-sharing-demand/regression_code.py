import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import statsmodels.api as sm

df = pd.read_csv("code/regression/bike-sharing-demand/train.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday

print(df.head())
df_model = df.drop(columns=["casual", "registered"])

print(df_model.columns)
df_model = df_model.drop(columns=["datetime"])
plt.figure(figsize=(8,5))

plt.hist(df["count"], bins=50)

plt.xlabel("Number of Bike Rentals")
plt.ylabel("Frequency")
plt.title("Distribution of Bike Rental Count")

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(
    df["temp"],
    df["count"],
    alpha=0.3
)

plt.xlabel("Temperature")
plt.ylabel("Bike Rental Count")
plt.title("Temperature vs Bike Rental Count")

plt.show()
X = df[["temp"]]
y = df["count"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

simple_model = LinearRegression()

simple_model.fit(X_train, y_train)

intercept = simple_model.intercept_
slope = simple_model.coef_[0]

print("Intercept =", intercept)
print("Slope =", slope)

print(
    f"Best-Fit Line:\n"
    f"y_hat = {intercept:.4f} + ({slope:.4f}) * temp"
)
y_train_pred = simple_model.predict(X_train)

plt.figure(figsize=(9,6))

plt.scatter(
    X_train["temp"],
    y_train,
    alpha=0.25,
    label="Observed values"
)

plt.plot(
    X_train["temp"],
    y_train_pred,
    linewidth=3,
    label="Best-fit line"
)

plt.xlabel("Temperature")
plt.ylabel("Bike Rental Count")
plt.title("Simple Linear Regression — Best-Fit Line")
plt.legend()

plt.show()

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": simple_model.predict(X_test)
})

results["Error"] = (
    results["Actual"] -
    results["Predicted"]
)

results["Absolute Error"] = abs(results["Error"])

print(results.head(10))

x_example = X_test.iloc[0]["temp"]
y_actual = y_test.iloc[0]
y_predicted = simple_model.predict(
    X_test.iloc[[0]]
)[0]

plt.figure(figsize=(8,6))

plt.scatter(
    X_train["temp"],
    y_train,
    alpha=0.2
)

plt.plot(
    X_train["temp"],
    y_train_pred,
    linewidth=3
)

plt.scatter(
    x_example,
    y_actual,
    s=100,
    label="Observed value"
)

plt.scatter(
    x_example,
    y_predicted,
    s=100,
    label="Predicted value"
)

plt.plot(
    [x_example, x_example],
    [y_actual, y_predicted],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Temperature")
plt.ylabel("Bike Rental Count")
plt.title("Observed Value, Prediction and Residual")

plt.legend()
plt.show()

print("Observed value =", y_actual)
print("Predicted value =", y_predicted)
print("Random error =", y_actual - y_predicted)
x = X_train["temp"].values
y_train_array = y_train.values

x_mean = np.mean(x)
y_mean = np.mean(y_train_array)

slope_ls = np.sum(
    (x - x_mean) * (y_train_array - y_mean)
) / np.sum(
    (x - x_mean) ** 2
)

intercept_ls = y_mean - slope_ls * x_mean

print("Least Squares Slope:", slope_ls)
print("Least Squares Intercept:", intercept_ls)

def hypothesis(X, theta0, theta1):
    return theta0 + theta1 * X
theta0 = intercept
theta1 = slope

sample_temperature = 20

prediction = hypothesis(
    sample_temperature,
    theta0,
    theta1
)

print("Temperature:", sample_temperature)
print("Predicted bike rentals:", prediction)

def compute_cost(X, y, theta0, theta1):

    predictions = theta0 + theta1 * X

    errors = predictions - y

    cost = (1 / (2 * len(y))) * np.sum(errors ** 2)

    return cost
cost = compute_cost(
    x,
    y_train_array,
    theta0,
    theta1
)

print("Cost =", cost)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = X_train_scaled.ravel()

y_train_gd = y_train.values
theta0 = 0.0
theta1 = 0.0

learning_rate = 0.01
iterations = 1000

cost_history = []

m = len(y_train_gd)

for i in range(iterations):

    predictions = theta0 + theta1 * X_train_scaled

    errors = predictions - y_train_gd

    gradient_theta0 = (1/m) * np.sum(errors)

    gradient_theta1 = (1/m) * np.sum(
        errors * X_train_scaled
    )

    theta0 -= learning_rate * gradient_theta0
    theta1 -= learning_rate * gradient_theta1

    cost = (1/(2*m)) * np.sum(errors ** 2)

    cost_history.append(cost)

print("Theta 0:", theta0)
print("Theta 1:", theta1)
plt.figure(figsize=(8,5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent — Cost vs Iteration")

plt.show()

features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "day",
    "hour",
    "weekday"
]

X = df[features]
y = df["count"]
X = pd.get_dummies(
    X,
    columns=["season", "weather"],
    drop_first=True
)

print(X.head())
print("\nNumber of features:", X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

multiple_model = LinearRegression()

multiple_model.fit(
    X_train,
    y_train
)

y_pred = multiple_model.predict(X_test)
mae = mean_absolute_error(
    y_test,
    y_pred
)

print("MAE:", mae)
mse = mean_squared_error(
    y_test,
    y_pred
)

print("MSE:", mse)
rmse = np.sqrt(mse)

print("RMSE:", rmse)
r2 = r2_score(
    y_test,
    y_pred
)

print("R²:", r2)
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (
    (1-r2) * (n-1) / (n-p-1)
)

print("Adjusted R²:", adjusted_r2)
metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2,
        adjusted_r2
    ]
})

print(metrics)
plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted Bike Rentals")

plt.show()
