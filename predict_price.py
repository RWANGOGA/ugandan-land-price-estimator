import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load the trained model
model = CatBoostRegressor()
model.load_model("uganda_land_price_model.cbm")

print("Uganda Land Price Predictor")
print("===========================\n")
print("Model loaded successfully!\n")


def predict_land_price(district, location, size_decimals, distance_km, electricity, water):
    """
    Predict land price using the trained CatBoost model
    """
    input_data = pd.DataFrame([{
        'district': district,
        'location': location,
        'log_size': np.log1p(size_decimals),
        'log_distance': np.log1p(distance_km),
        'electricity': 1 if electricity == "Yes" else 0,
        'water': 1 if water == "Yes" else 0
    }])

    log_pred = model.predict(input_data)
    price = np.expm1(log_pred)[0]
    
    return f"Estimated land price: UGX {price:,.0f}"


def get_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ['yes', 'y', 'true', '1']:
            return "Yes"
        if ans in ['no', 'n', 'false', '0']:
            return "No"
        print("Please answer Yes or No")


# ────────────────────────────────────────────────
# Main interactive loop
# ────────────────────────────────────────────────
while True:
    print("\nEnter land details (or type 'exit' to quit)\n")

    district = input("District (e.g. Wakiso, Mukono, Kampala): ").strip()
    if district.lower() == 'exit':
        print("\nThank you for using the Uganda Land Price Predictor!")
        break

    location = input("Specific location/area (e.g. Kira - Kimwanyi): ").strip()

    try:
        size = float(input("Size in decimals (e.g. 25): "))
        if size <= 0:
            raise ValueError
    except:
        print("Invalid or negative number → using default 20 decimals.")
        size = 20.0

    try:
        distance = float(input("Distance to main tarmac road in km (e.g. 2.5): "))
        if distance < 0:
            raise ValueError
    except:
        print("Invalid or negative number → using default 3 km.")
        distance = 3.0

    electricity = get_yes_no("Electricity available? (Yes/No): ")
    water = get_yes_no("Water available? (Yes/No): ")

    # Make prediction
    result = predict_land_price(district, location, size, distance, electricity, water)
    
    print("\n" + "="*70)
    print(result)
    print("="*70)