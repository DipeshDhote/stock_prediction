from pathlib import Path
import tensorflow as tf
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Import custom functions
from src.components.model_prediction import predict_stock
from src.components.data_ingestion import load_data

# Streamlit UI
st.title("Welcome to Stock Prediction Project")

stock_symbols = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
    "KOTAKBANK", "LT", "SBIN", "BAJFINANCE", "ITC", "ASIANPAINT",
    "HCLTECH", "AXISBANK"
]

selected_stock = st.sidebar.selectbox("Pick a Stock:", stock_symbols, key="stock_dropdown")

# Load Scaler with caching
@st.cache_resource
def load_scaler():
    return joblib.load(Path("models") / "scaler.joblib")

scaler = load_scaler()

# Define and load model
model_path = Path(f"models/{selected_stock}_model.h5")

@st.cache_resource
def load_model(path):
    if path.exists():
        return tf.keras.models.load_model(path)
    else:
        st.error(f"Model file not found: {path}")
        st.stop()

stock_model = load_model(model_path)

# Load Stock Data
stock_data = load_data(selected_stock)

# Predict button
if st.button("Predict"):
    with st.spinner("Predicting... Please wait!"):
        # Get Prediction
        prediction, dates = predict_stock(
            model=stock_model, df=stock_data, scaler=scaler,
            feature="Close", lookback=60, future_days=30
        )

    # Flatten and round predictions
    predictions = [round(float(i), 2) for i in np.array(prediction).flatten()]

    # Create DataFrame for visualization
    data = pd.DataFrame({"Date": dates, "Prediction": predictions})
    data.set_index("Date", inplace=True)

    # Plot future predictions
    # Create a line plot
    fig = px.line(data, x= data.index, y= "Prediction", title="Future Stock Price Prediction (30 Days)",
                markers=True, template="plotly_dark")
    


    # Show the plot 
    st.plotly_chart(fig)



