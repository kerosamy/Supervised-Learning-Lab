import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------
# 1. Load the dataset
# ------------------------
data = pd.read_csv("California_Houses.csv")

# ------------------------
# 2. Split features and target
# ------------------------
X = data.drop("Median_House_Value", axis=1)
y = data["Median_House_Value"]

# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ------------------------
# 3. Feature scaling
# ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ------------------------
# 4. Train Linear Regression
# ------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# ------------------------
# 5. Loop through different alpha for Ridge and Lasso
# ------------------------
alphas = [0.01, 0.1, 1, 10, 100]

print("Ridge Regression Results:")
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Alpha={alpha} -> MSE: {mse:.2f}, R²: {r2:.3f}")

print("\nLasso Regression Results:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Alpha={alpha} -> MSE: {mse:.2f}, R²: {r2:.3f}")

# ------------------------
# 6. Evaluate Linear Regression
# ------------------------
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f"\nLinear Regression -> MSE: {mse_lin:.2f}, R²: {r2_lin:.3f}")

# ------------------------
# 7. Optional: Visualize Ridge predictions with best alpha
# ------------------------
best_alpha = 1  # replace with best from loop
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train_scaled, y_train)
y_pred_best = ridge_best.predict(X_test_scaled)



from sklearn.linear_model import SGDRegressor

# SGD Regressor - Batch (simulate batch gradient with large batch = all samples)
sgd_batch = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42)
sgd_batch.fit(X_train_scaled, y_train)
y_pred_sgd_batch = sgd_batch.predict(X_test_scaled)

# SGD Regressor - Stochastic
sgd_stochastic = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_stochastic.fit(X_train_scaled, y_train)
y_pred_sgd_stochastic = sgd_stochastic.predict(X_test_scaled)

# Evaluate
def evaluate_sgd(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.3f}")

evaluate_sgd(y_test, y_pred_sgd_batch, "SGD Batch")
evaluate_sgd(y_test, y_pred_sgd_stochastic, "SGD Stochastic")


plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(f"Ridge Regression (alpha={best_alpha}): Actual vs Predicted")
plt.show()
