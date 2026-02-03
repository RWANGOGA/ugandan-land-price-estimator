import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime
import os
import folium
from streamlit_folium import st_folium
import requests

# ────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────
MODEL_PATH = "uganda_land_price_model.cbm"
QUERIES_FILE = "user_queries.csv"

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# ────────────────────────────────────────────────
# Load and CLEAN training data
# ────────────────────────────────────────────────
@st.cache_data
def load_training_data():
    return pd.read_csv("uganda_land_prices.csv")  # Update filename if different

# Load the data FIRST
df_train = load_training_data()

# THEN clean it
df_train['price_ugx'] = pd.to_numeric(df_train['price_ugx'], errors='coerce')

# Remove any rows where price could not be converted
df_train = df_train.dropna(subset=['price_ugx'])

# Make sure it's integer type
df_train['price_ugx'] = df_train['price_ugx'].astype('int64')

# Print info to terminal (useful for debugging)
st.write(f"Loaded and cleaned: {len(df_train)} rows")
# You can also see this in the terminal when running the app

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
    df_new = pd.DataFrame([query_data])
    df_new.to_csv(QUERIES_FILE, mode='a', header=not file_exists, index=False)

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
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon, data[0].get('display_name', query)
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
# Coordinates (expanded)
# ────────────────────────────────────────────────
location_coords = {
    "Kira - Kimwanyi": {"lat": 0.366, "lon": 32.635},
    "Namugongo - Sonde": {"lat": 0.362, "lon": 32.675},
    "Seeta - Bajjo": {"lat": 0.380, "lon": 32.745},
    "Buziga": {"lat": 0.280, "lon": 32.620},
    "Kololo": {"lat": 0.320, "lon": 32.590},
    "Gayaza - Kiwenda": {"lat": 0.430, "lon": 32.615},
    "Kasangati - Nangabo": {"lat": 0.420, "lon": 32.600},
    "Kitende - along Entebbe Road": {"lat": 0.210, "lon": 32.560},
    "Kabembe - off Kayunga Road": {"lat": 0.410, "lon": 32.780},
    "Ntoroko Town": {"lat": 0.8833, "lon": 30.5333},
    "Kazo Town": {"lat": -0.0500, "lon": 30.7500},
    "Bundibugyo Town": {"lat": 0.7111, "lon": 30.0647},
    "Kamwenge Town": {"lat": 0.1889, "lon": 30.4539},
    "Gulu Town": {"lat": 2.7714, "lon": 32.2990},
    "Kitgum Town": {"lat": 3.2789, "lon": 32.8867},
    "Lira Town": {"lat": 2.2350, "lon": 32.9097},
    # Add more as needed
}

district_centers = {
    "Wakiso": {"lat": 0.400, "lon": 32.500},
    "Mukono": {"lat": 0.353, "lon": 32.755},
    "Kampala": {"lat": 0.313, "lon": 32.581},
    "Mpigi": {"lat": 0.230, "lon": 32.320},
    "Luweero": {"lat": 0.830, "lon": 32.500},
    "Ntoroko": {"lat": 0.8833, "lon": 30.5333},
    "Kazo": {"lat": -0.0500, "lon": 30.7500},
    "Bundibugyo": {"lat": 0.7111, "lon": 30.0647},
    "Kamwenge": {"lat": 0.1889, "lon": 30.4539},
    "Gulu": {"lat": 2.7714, "lon": 32.2990},
    "Kitgum": {"lat": 3.2789, "lon": 32.8867},
    "Lira": {"lat": 2.2350, "lon": 32.9097},
}

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ────────────────────────────────────────────────
# Streamlit App (Updated Layout)
# ────────────────────────────────────────────────
st.set_page_config(page_title="Uganda Land Price Estimator", layout="wide")

# Sidebar for inputs
with st.sidebar:
    st.header("Enter Land Details")
    district = st.text_input("District (e.g. Wakiso, Gulu, Ntoroko)", value="Wakiso")
    location = st.text_input("Location (e.g. Kira - Kimwanyi, Gulu Town)", value="Kira - Kimwanyi")
    size_dec = st.number_input("Size (decimals)", min_value=1.0, value=25.0, step=0.5)
    distance_km = st.number_input("Distance to tarmac (km)", min_value=0.0, value=2.0, step=0.1)
    electricity = st.radio("Electricity", ["Yes", "No"], index=0)
    water = st.radio("Water", ["Yes", "No"], index=0)
    
    if st.button("Estimate Price", type="primary"):
        if district and location:
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
        else:
            st.error("Fill District and Location.")
    
    if st.button("Clear Results"):
        st.session_state.prediction_results = None

# Main area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediction", "Map & Stats", "Trends", "Similar Lands", "Validate Price"])

if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    district = results['district']
    location = results['location']
    size_dec = results['size_dec']
    distance_km = results['distance_km']
    electricity = results['electricity']
    water = results['water']
    price = results['price']

    # Confidence & range
    loc_clean = location.lower()
    count = location_counts.get(location, 0) + location_counts.get(loc_clean, 0)
    base_range_low = 0.85 - (distance_km * 0.02)
    base_range_high = 1.15 + (distance_km * 0.02)

    if count >= 8:
        confidence = "High"
        range_pct_low = max(0.75, base_range_low - 0.05)
        range_pct_high = min(1.30, base_range_high + 0.05)
    elif count >= 3:
        confidence = "Medium"
        range_pct_low = max(0.70, base_range_low - 0.10)
        range_pct_high = min(1.40, base_range_high + 0.10)
    else:
        confidence = "Low (uncommon location)"
        range_pct_low = max(0.60, base_range_low - 0.15)
        range_pct_high = min(1.55, base_range_high + 0.15)

    lower = price * range_pct_low
    upper = price * range_pct_high

    with tab1:
        st.success("Estimate ready!")
        st.markdown(f"### Estimated Value: UGX {price:,.0f}")
        st.markdown(f"**Range:** UGX {lower:,.0f} – {upper:,.0f}")
        st.caption(f"Confidence: {confidence}")
        with st.expander("Why this price?"):
            st.markdown("""
            - Location: Central higher than upcountry
            - Size: Larger = lower per decimal
            - Utilities: Big boost in rural areas
            - Distance: Critical in upcountry
            """)
            if electricity == "Yes" and water == "Yes":
                st.info("Utilities boost value")
            elif electricity == "No" and water == "No":
                st.warning("No utilities lower value")
        st.success("Query saved!")

    with tab2:
        st.subheader("Map & Stats")
        lat, lon, display_name = get_coordinates(location, district)
        if lat and lon:
            m = folium.Map(location=[lat, lon], zoom_start=14)
            folium.Marker([lat, lon], popup=display_name or location).add_to(m)
            st_folium(m, width=700, height=400, key=f"map_{location}_{district}")
        else:
            dist_lower = district.lower()
            if dist_lower in district_centers:
                coords = district_centers[dist_lower]
                m = folium.Map(location=[coords['lat'], coords['lon']], zoom_start=11)
                folium.Marker([coords['lat'], coords['lon']], popup=district).add_to(m)
                st_folium(m, width=700, height=400, key=f"fallback_map_{district}")
            else:
                st.info("No map data.")
        population = get_population(district)
        st.markdown(f"**Population:** {population}")
        st.link_button("Google Maps Search (Businesses/Schools)", f"https://www.google.com/maps/search/?api=1&query=businesses+schools+{location}+{district}+Uganda")

    with tab3:
        st.subheader("Trends per District")
        avg_price_district = df_train.groupby('district')['price_ugx'].mean().sort_values(ascending=False)
        avg_price_per_decimal = avg_price_district / df_train.groupby('district')['size_decimals'].mean()
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(avg_price_district, use_container_width=True)
        with col2:
            st.bar_chart(avg_price_per_decimal, use_container_width=True)

    with tab4:
        st.subheader("Similar Lands")
        similar_lands = df_train[
            (df_train['district'].str.lower() == district.lower()) &
            (df_train['location'].str.lower().str.contains(location.lower()))
        ].head(5)
        if not similar_lands.empty:
            st.dataframe(similar_lands[['location', 'size_decimals', 'distance_km', 'electricity', 'water', 'price_ugx']])
        else:
            st.info("No matches. Check district averages above.")

    with tab5:
        st.subheader("Validate Your Price")
        user_price = st.number_input("Your expected price (UGX)", min_value=1000000.0, step=1000000.0)
        if st.button("Validate"):
            if user_price > price * 1.5:
                st.error(f"Exceeds estimate significantly. Cannot proceed.")
            elif user_price < price * 0.5:
                st.warning("Much lower than estimate. Undervaluing?")
            else:
                st.success("Within range. Proceed.")

# Footer
st.markdown("---")
st.caption("For informational use. Upcountry data limited – results approximate.")