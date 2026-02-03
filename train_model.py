import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# =====================================
# 1. Load and prepare the data
# =====================================
df = pd.read_csv("uganda_land_prices.csv")

# Convert Yes/No → 1/0
df['electricity'] = df['electricity'].map({'Yes': 1, 'No': 0})
df['water']       = df['water'].map({'Yes': 1, 'No': 0})

# Log transformations
df['log_price']      = np.log1p(df['price_ugx'])
df['log_size']       = np.log1p(df['size_decimals'])
df['log_distance']   = np.log1p(df['distance_km'])

# Fill any missing (shouldn't be any, but safe)
df['log_distance'] = df['log_distance'].fillna(df['log_distance'].median())

# =====================================
# 2. Features and target
# =====================================
features = ['district', 'location', 'log_size', 'log_distance', 'electricity', 'water']
target   = 'log_price'

X = df[features]
y = df[target]

# =====================================
# 3. Train / Test split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,          # 20% for testing
    random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# =====================================
# 4. Train CatBoost model
# =====================================
cat_features = ['district', 'location', 'electricity', 'water']

model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    loss_function='RMSE',
    verbose=200              # show progress every 200 iterations
)

print("\nTraining model... please wait a moment\n")

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=80
)

# =====================================
# 5. Make predictions and evaluate
# =====================================
y_pred_log = model.predict(X_test)

# Convert back from log to real UGX prices
y_pred_real  = np.expm1(y_pred_log)
y_test_real  = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred_real)
r2  = r2_score(y_test_real, y_pred_real)

print("\n" + "="*50)
print("Model Performance on Test Set")
print("="*50)
print(f"MAE (Mean Absolute Error):     UGX {mae:,.0f}")
print(f"R² score:                       {r2:.4f}")
print(f"Median price in test set:       UGX {np.median(y_test_real):,.0f}")
print("="*50)

# Show a few example predictions
print("\nSome example predictions vs actual:")
examples = pd.DataFrame({
    'District': X_test['district'].values[:5],
    'Location': X_test['location'].values[:5],
    'Size (dec)': np.expm1(X_test['log_size'].values[:5]).astype(int),
    'Actual (UGX)': y_test_real[:5].astype(int),
    'Predicted (UGX)': y_pred_real[:5].astype(int)
})
# =====================================
# 6. Save the model so you can use it later
# =====================================
model.save_model("uganda_land_price_model.cbm")

print("\nModel successfully saved as: uganda_land_price_model.cbm")
print("You can now load it anytime without retraining!")
print(examples)