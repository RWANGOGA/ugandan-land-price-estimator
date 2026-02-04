import streamlit as st

st.title("Validate Your Expected Price")

if 'prediction_results' in st.session_state:
    price = st.session_state.prediction_results['price']

    user_price = st.number_input("Enter your expected price (UGX)", min_value=1000000.0, step=1000000.0)

    if st.button("Validate"):
        if user_price > price * 1.5:
            st.error(f"Your price (UGX {user_price:,.0f}) exceeds the estimated value significantly. Cannot proceed.")
        elif user_price < price * 0.5:
            st.warning(f"Your price is much lower than estimated. You might be undervaluing the land.")
        else:
            st.success(f"Your price (UGX {user_price:,.0f}) is within reasonable range. Good to proceed.")
else:
    st.info("Make a prediction first from the main page.")