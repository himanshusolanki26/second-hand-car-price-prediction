import streamlit as st
import numpy as np
import joblib

# Load the model, scaler, and columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# App Configuration
st.set_page_config(page_title="Second-Hand Car Price Prediction", page_icon="🚗", layout="wide")

# ---- HEADER ----
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-size: 38px;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #7f8c8d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">🚗 Second-Hand Car Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate the **resale value** of a second-hand car in <b>USD ($)</b></p>', unsafe_allow_html=True)

# ---- SIDEBAR INPUT ----
st.sidebar.header("🔧 Enter Car Details")

st.sidebar.markdown("This tool predicts the **resale value** of a used (second-hand) car.")

# Numerical Inputs
year = st.sidebar.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1, value=2018)
mileage = st.sidebar.number_input("Mileage (in miles)", min_value=0, step=1000, value=30000)
tax = st.sidebar.number_input("Tax (in $)", min_value=0, step=10, value=150)
mpg = st.sidebar.number_input("Miles Per Gallon (MPG)", min_value=0.0, step=0.1, value=50.0)
engine_size = st.sidebar.number_input("Engine Size (L)", min_value=0.0, step=0.1, value=1.5)

# Dropdowns for categorical variables
model_options = [col.replace("model_ ", "") for col in columns if col.startswith("model_ ")]
selected_model = st.sidebar.selectbox("Car Model", model_options)

transmission_options = ["Manual", "Semi-Auto"]
selected_transmission = st.sidebar.selectbox("Transmission", transmission_options)

fuel_options = ["Electric", "Hybrid", "Other", "Petrol"]
selected_fuel = st.sidebar.selectbox("Fuel Type", fuel_options)

# ---- PROCESS INPUT ----
input_data = np.zeros(len(columns))
input_data[columns.index("year")] = year
input_data[columns.index("mileage")] = mileage
input_data[columns.index("tax")] = tax
input_data[columns.index("mpg")] = mpg
input_data[columns.index("engineSize")] = engine_size

if f"model_ {selected_model}" in columns:
    input_data[columns.index(f"model_ {selected_model}")] = 1
if f"transmission_{selected_transmission}" in columns:
    input_data[columns.index(f"transmission_{selected_transmission}")] = 1
if f"fuelType_{selected_fuel}" in columns:
    input_data[columns.index(f"fuelType_{selected_fuel}")] = 1

# Scale the full feature vector
scaled_input = scaler.transform([input_data])[0]

# ---- PREDICTION OUTPUT ----
st.markdown("---")
if st.button("🔮 Predict Second-Hand Car Price"):
    prediction = model.predict([scaled_input])[0]

    st.markdown(
        f"""
        <div style="background-color:#eafaf1;padding:25px;border-radius:15px;text-align:center;">
            <h2 style="color:#1abc9c;">💰 Estimated Second-Hand Car Price:</h2>
            <h1 style="color:#16a085;">${prediction:,.2f}</h1>
            <p style="color:#2c3e50;font-size:16px;">(Approximate resale value based on current market trends)</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("👈 Enter the details in the sidebar and click **Predict Second-Hand Car Price**.")

# ---- FOOTER ----
st.markdown("<br><hr><p style='text-align:center; color:gray;'>🔑 Tip: Older cars with high mileage usually have lower resale value.<br>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
