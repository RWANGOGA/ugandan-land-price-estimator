import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("Map & Area Statistics")

# Reuse variables from session state
if 'prediction_results' in st.session_state:
    results = st.session_state.prediction_results
    district = results['district']
    location = results['location']

    # Map
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
            st.info("No map data available.")

    # Population & Google Maps link
    population = get_population(district)
    st.markdown(f"**Population (approx.):** {population}")
    st.link_button(
        "Search Businesses / Schools on Google Maps",
        f"https://www.google.com/maps/search/?api=1&query=businesses+schools+{location}+{district}+Uganda"
    )
else:
    st.info("Please make a prediction first from the main page.")