import streamlit as st

st.title("Similar Lands from Recent Data")

if 'prediction_results' in st.session_state and 'df_train' in globals():
    results = st.session_state.prediction_results
    district = results['district']
    location = results['location']
    loc_clean = location.lower()

    similar_lands = df_train[
        (df_train['district'].str.lower() == district.lower()) &
        (df_train['location'].str.lower().str.contains(loc_clean))
    ].head(5)

    if not similar_lands.empty:
        st.dataframe(
            similar_lands[['location', 'size_decimals', 'distance_km', 'electricity', 'water', 'price_ugx']],
            column_config={"price_ugx": st.column_config.NumberColumn(format="UGX %d")},
            use_container_width=True
        )
    else:
        st.info("No similar lands found in recent data for this location.")
else:
    st.info("Make a prediction first from the main page.")