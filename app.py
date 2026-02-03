import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime
import os
import folium
from streamlit_folium import st_folium
import requests
import urllib.request   # ✅ ADDED

# ────────────────────────────────────────────────
# Streamlit config (must be early)
# ────────────────────────────────────────────────
st.set_page_config(page_title="Uganda Land Price Estimator", layout="wide")

# ────────────────────────────────────────────────
# Paths & Model URL
# ────────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CVZH0RVKKrenuxJuq9UMdaO47wOiJOGD"
MODEL_PATH = "uganda_land_price_model.cbm"
QUERIES_FILE = "user_queries.csv"

# ────────────────────────────────────────────────
# Load model (AUTO DOWNLOAD)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading land price model... (first time only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")

    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# ────────────────────────────────────────────────
# Load and CLEAN training data
# ────────────────────────────────────────────────
@st.cache_data
def load_training_data():
    return pd.read_csv("uganda_land_prices.csv")

df_train = load_training_data()

df_train['price_ugx'] = pd.to_numeric(df_train['price_ugx'], errors='coerce')
df_train = df_train.dropna(subset=['price_ugx'])
df_train['price_ugx'] = df_train['price_ugx'].astype('int64')

st.write(f"Loaded and cleaned: {len(df_train)} rows")

location_counts = df_train['location'].value_counts().to_dict()

# ────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────
def predict_land_price(district, location, size_decimals, distance_km, electricity, water):
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

# ────────────────────────────────────────────────
# Save query
# ────────────────────────────────────────────────
def save_query(district, location, size, distance, electricity, water, predicted_price):
    query_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'district': district,
        'location': location,
        'size_decimals': size,
        'distance_km': distance,
        'electricity': electricity,
        'water': water,
        'predicted_price_ugx': round(predicted_price),
    }
    file_exists = os.path.isfile(QUERIES_FILE)
    pd.DataFrame([query_data]).to_csv(
        QUERIES_FILE, mode='a', header=not file_exists, index=False
    )

# ────────────────────────────────────────────────
# Geocode
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_coordinates(location, district):
    query = f"{location}, {district}, Uganda"
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}"
    headers = {'User-Agent': 'UgandaLandPriceEstimator/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', query)
    except:
        pass
    return None, None, None

# ────────────────────────────────────────────────
# Population
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_population(district):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&exsentences=2&format=json&titles={district}_District"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        page_id = list(data['query']['pages'].keys())[0]
        extract = data['query']['pages'][page_id].get('extract', '')
        if 'population' in extract.lower():
            start = extract.lower().find('population')
            end = extract.find('.', start)
            return extract[start:end+1]
    except:
        pass
    return "Population data not available."

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Enter Land Details")
    district = st.text_input("District", value="Wakiso")
    location = st.text_input("Location", value="Kira - Kimwanyi")
    size_dec = st.number_input("Size (decimals)", min_value=1.0, value=25.0, step=0.5)
    distance_km = st.number_input("Distance to tarmac (km)", min_value=0.0, value=2.0, step=0.1)
    electricity = st.radio("Electricity", ["Yes", "No"], index=0)
    water = st.radio("Water", ["Yes", "No"], index=0)

    if st.button("Estimate Price", type="primary"):
        with st.spinner("Estimating..."):
            price = predict_land_price(district, location, size_dec, distance_km, electricity, water)
            save_query(district, location, size_dec, distance_km, electricity, water, price)
            st.session_state.prediction_results = {
                'district': district,
                'location': location,
                'size_dec': size_dec,
                'distance_km': distance_km,
                'electricity': electricity,
                'water': water,
                'price': price
            }

    if st.button("Clear Results"):
        st.session_state.prediction_results = None

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Prediction", "Map & Stats", "Trends", "Similar Lands", "Validate Price"]
)

if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    price = results['price']

    with tab1:
        st.success("Estimate ready!")
        st.markdown(f"### Estimated Value: UGX {price:,.0f}")
        st.success("Query saved!")

# Footer
st.markdown("---")
st.caption("For informational use. Upcountry data limited – results approximate.")
