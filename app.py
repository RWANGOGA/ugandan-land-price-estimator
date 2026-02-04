import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime
import os
import folium
from streamlit_folium import st_folium
import requests
import urllib.request

# ────────────────────────────────────────────────
# Page config (must be first command)
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Uganda Land Price Estimator",
    page_icon="🇺🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Paths & Model URL
# ────────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CVZH0RVKKrenuxJuq9UMdaO47wOiJOGD"
MODEL_PATH = "uganda_land_price_model.cbm"
QUERIES_FILE = "user_queries.csv"

# ────────────────────────────────────────────────
# Load model with auto-download
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model (one-time process)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded!")

        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# ────────────────────────────────────────────────
# Load & clean training data
# ────────────────────────────────────────────────
@st.cache_data
def load_training_data():
    csv_path = "uganda_land_prices.csv"
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['price_ugx'] = pd.to_numeric(df['price_ugx'], errors='coerce')
        df = df.dropna(subset=['price_ugx'])
        df['price_ugx'] = df['price_ugx'].astype('int64')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

df_train = load_training_data()

if df_train is not None:
    location_counts = df_train['location'].value_counts().to_dict()
else:
    location_counts = {}

# ────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────
def predict_land_price(district, location, size_decimals, distance_km, electricity, water):
    if model is None:
        return None
    
    try:
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
    except:
        return None

# ────────────────────────────────────────────────
# Save query
# ────────────────────────────────────────────────
def save_query(district, location, size, distance, electricity, water, predicted_price):
    if predicted_price is None:
        return
    try:
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
        pd.DataFrame([query_data]).to_csv(QUERIES_FILE, mode='a', header=not file_exists, index=False)
    except:
        pass

# ────────────────────────────────────────────────
# Geocode & Population (keep your existing functions)
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_coordinates(location, district):
    query = f"{location}, {district}, Uganda"
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}"
    headers = {'User-Agent': 'UgandaLandPriceEstimator/1.0'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', query)
    except:
        pass
    return None, None, None

@st.cache_data(ttl=3600)
def get_population(district):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&exsentences=2&format=json&titles={district}_District"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        page_id = list(data['query']['pages'].keys())[0]
        extract = data['query']['pages'][page_id].get('extract', '')
        if 'population' in extract.lower():
            start = extract.lower().find('population')
            end = extract.find('.', start)
            return extract[start:end+1].strip()
    except:
        pass
    return "Not available"

# District centers (fallback)
district_centers = {
    "Wakiso": {"lat": 0.400, "lon": 32.500},
    "Mukono": {"lat": 0.353, "lon": 32.755},
    "Kampala": {"lat": 0.313, "lon": 32.581},
    # Add more as needed
}

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ────────────────────────────────────────────────
# Sidebar – Input form
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Land Details")
    district = st.text_input("District", value="Wakiso")
    location = st.text_input("Location", value="Kira - Kimwanyi")
    size_dec = st.number_input("Size (decimals)", min_value=1.0, value=25.0, step=0.5)
    distance_km = st.number_input("Distance to tarmac (km)", min_value=0.0, value=2.0, step=0.1)
    electricity = st.radio("Electricity", ["Yes", "No"], index=0)
    water = st.radio("Water", ["Yes", "No"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Estimate", type="primary"):
            if model is None or df_train is None:
                st.error("Model or data not ready.")
            else:
                with st.spinner("Estimating..."):
                    price = predict_land_price(district, location, size_dec, distance_km, electricity, water)
                    if price is not None:
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
    with col2:
        if st.button("Clear"):
            st.session_state.prediction_results = None

# ────────────────────────────────────────────────
# Main content – Tabs
# ────────────────────────────────────────────────
if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    price = results['price']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Prediction Result",
        "Map & Stats",
        "Trends",
        "Similar Lands",
        "Validate Price"
    ])

    # Tab 1: Prediction Result
    with tab1:
        st.success("Estimate ready!")
        st.markdown(f"### Estimated Value")
        st.markdown(f"<h2 style='color:#2e7d32;'>UGX {price:,.0f}</h2>", unsafe_allow_html=True)
        st.success("Query saved!")

    # Tab 2: Map & Stats
    with tab2:
        st.subheader("Approximate Location Map")
        lat, lon, display_name = get_coordinates(results['location'], results['district'])
        if lat and lon:
            m = folium.Map(location=[lat, lon], zoom_start=14)
            folium.Marker([lat, lon], popup=display_name or results['location']).add_to(m)
            st_folium(m, width=700, height=400)
        else:
            st.caption("Showing district center (exact location not found)")
            dist = results['district'].lower()
            if dist in district_centers:
                coords = district_centers[dist]
                m = folium.Map(location=[coords['lat'], coords['lon']], zoom_start=11)
                folium.Marker([coords['lat'], coords['lon']], popup=results['district']).add_to(m)
                st_folium(m, width=700, height=400)

        st.subheader("Area Statistics")
        population = get_population(results['district'])
        st.markdown(f"**Population (approx.):** {population}")
        st.link_button(
            "Search Businesses / Schools on Google Maps",
            f"https://www.google.com/maps/search/?api=1&query=businesses+schools+{results['location']}+{results['district']}+Uganda"
        )

    # Tab 3: Trends
    with tab3:
        st.subheader("Historical Price Trends per District")
        if df_train is not None:
            avg_price_district = df_train.groupby('district')['price_ugx'].mean().sort_values(ascending=False)
            avg_price_per_decimal = avg_price_district / df_train.groupby('district')['size_decimals'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Avg Total Price (UGX)**")
                st.bar_chart(avg_price_district)
            with col2:
                st.markdown("**Avg Price per Decimal (UGX)**")
                st.bar_chart(avg_price_per_decimal)
        else:
            st.warning("No trend data available.")

    # Tab 4: Similar Lands
    with tab4:
        st.subheader("Similar Lands from Recent Data")
        if df_train is not None:
            loc_clean = results['location'].lower()
            similar = df_train[
                (df_train['district'].str.lower() == results['district'].lower()) &
                (df_train['location'].str.lower().str.contains(loc_clean))
            ].head(5)

            if not similar.empty:
                st.dataframe(
                    similar[['location', 'size_decimals', 'distance_km', 'electricity', 'water', 'price_ugx']],
                    column_config={"price_ugx": st.column_config.NumberColumn(format="UGX %d")}
                )
            else:
                st.info("No similar lands found for this location.")
        else:
            st.warning("No data available.")

    # Tab 5: Validate Price
    with tab5:
        st.subheader("Validate Your Expected Price")
        user_price = st.number_input("Your expected price (UGX)", min_value=1000000.0, step=1000000.0)
        if st.button("Check"):
            if user_price > price * 1.5:
                st.error(f"Too high (exceeds estimate by more than 50%). Adjust or re-check details.")
            elif user_price < price * 0.5:
                st.warning("Quite low compared to estimate. You might be undervaluing.")
            else:
                st.success(f"Reasonable range. Good to proceed.")
else:
    # When no prediction yet
    st.info("Please fill the sidebar form and click 'Estimate' to see results.")