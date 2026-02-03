from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

app = Flask(__name__)
model = CatBoostRegressor()
model.load_model("uganda_land_price_model.cbm")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        district = request.form['district']
        location = request.form['location']
        size = float(request.form['size'])
        distance = float(request.form['distance'])
        electricity = "Yes" if 'electricity' in request.form else "No"
        water = "Yes" if 'water' in request.form else "No"
        price = predict_land_price(district, location, size, distance, electricity, water)
        return render_template('index.html', price=price)
    return render_template('index.html', price=None)

def predict_land_price(district, location, size_decimals, distance_km, electricity, water):
    # Same as before
    input_data = pd.DataFrame([{
        'district': district,
        'location': location,
        'log_size': np.log1p(size_decimals),
        'log_distance': np.log1p(distance_km),
        'electricity': 1 if electricity == "Yes" else 0,
        'water': 1 if water == "Yes" else 0
    }])
    log_pred = model.predict(input_data)
    return np.expm1(log_pred)[0]

if __name__ == '__main__':
    app.run(debug=True)