import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 4: Predict Future Prices
def predict_stock(model, df, scaler, feature="Close", lookback=60, future_days=30):
    # Check if DataFrame is valid
    if df.empty or len(df) < lookback:
        raise ValueError("Insufficient data for prediction. Check DataFrame length.")

    # Extract and scale the feature column
    data = df[[feature]].values
    scaled_data = scaler.transform(data.reshape(-1, 1))  # Ensure correct shape

    # Prepare test data
    X_test = []
    for i in range(lookback, len(scaled_data)):
        X_test.append(scaled_data[i-lookback:i, 0])

    X_test = np.array(X_test).reshape(-1, lookback, 1)

    # Predict prices
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot actual vs. predicted prices
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index[lookback:], df[feature][lookback:], label="Actual Prices", color='blue')
    # plt.plot(df.index[lookback:], predicted_prices, label="Predicted Prices", color='red')
    # plt.title("Stock Price Prediction")
    # plt.xlabel("Date")
    # plt.ylabel("Stock Price")
    # plt.legend()
    

    # Future Prediction
    last_data = scaled_data[-lookback:].reshape(1, lookback, 1)

    future_predictions = []
    for _ in range(future_days):
        next_pred = model.predict(last_data)  # Predict next step
        future_predictions.append(next_pred[0, 0])

        # Update last_data for next prediction
        last_data = np.append(last_data[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    # Convert predictions back to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate future dates
    future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1)[1:]

    # Plot future predictions
    # plt.figure(figsize=(12, 6))
    # plt.plot(future_dates, future_predictions, marker='o', label="Future Predictions", color="green")
    # plt.title(f"Future Stock Price Prediction ({future_days} Days)")
    # plt.xlabel("Date")
    # plt.ylabel("Predicted Price")
    # plt.legend()


    return future_predictions,future_dates
