import streamlit as st

st.title("Historical Price Trends per District")

if 'df_train' in globals():
    avg_price_district = df_train.groupby('district')['price_ugx'].mean().sort_values(ascending=False)
    avg_price_per_decimal = avg_price_district / df_train.groupby('district')['size_decimals'].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Total Price (UGX)**")
        st.bar_chart(avg_price_district, use_container_width=True)

    with col2:
        st.markdown("**Average Price per Decimal (UGX)**")
        st.bar_chart(avg_price_per_decimal, use_container_width=True)
else:
    st.warning("Training data not loaded.")