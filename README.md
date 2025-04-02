# Stock Price Prediction using LSTM

## 📌 Project Overview
This project focuses on Time Series Forecasting for predicting the stock prices of NIFTY stocks using Long Short-Term Memory (LSTM) networks. The goal is to build a deep learning model that can capture historical price patterns and make future stock price predictions.

## 🛠 Tech Stack
- **Python** (Programming Language)
- **TensorFlow/Keras** (Deep Learning Framework)
- **Pandas, NumPy** (Data Handling)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-Learn** (Preprocessing & Evaluation)
- **Yahoo Finance API** (Stock Data Extraction)

## 📊 Dataset
- **Source:** Yahoo Finance API
- **Features Used:**
  - Open Price
  - High Price
  - Low Price
  - Close Price
  - Volume
  - Adjusted Close Price
- **Time Period:** Last 10+ years of historical data for NIFTY 50 stocks

## 🔍 Data Preprocessing
- **Data Cleaning:** Handling missing values
- **Normalization:** Using MinMaxScaler
- **Creating Sequences:** Preparing data for LSTM input
- **Splitting Data:** Dividing into Training and Testing sets

## 🏗️ Model Architecture
LSTM Network with the following layers:
- **Input Layer:** Time Steps & Features
- **LSTM Layers**
- **Dropout Layers:** Prevent Overfitting
- **Dense Layer:** Final Prediction Output
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam Optimizer

## 🚀 Training and Evaluation
- **Training:** Model is trained on past stock price data
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
- **Visualization:** Plot actual vs predicted stock prices

## 📈 Results
- The model successfully learns trends and patterns in stock price movement.
- Predicted prices closely align with real prices, but stock market volatility makes long-term forecasting challenging.

## 🔧 How to Run the Project
1. **Clone the repository:**
   ```sh
   git clone https://github.com/DipeshDhote/stock_price_prediction.git
   cd nifty50-lstm
   ```
2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the model:**
   ```sh
   python train.py
   ```
4. **Make predictions:**
   ```sh
   python predict.py
   ```

## 📂 Project Structure
```
├── data/                  # Dataset Files
├── models/                # Saved LSTM Model
├── notebook/             # Jupyter Notebooks for EDA & Training
├── src/
│   ├── data_preprocessing.py  # Data Cleaning & Feature Engineering
│   ├── model.py               # LSTM Model Implementation
│   ├── train.py               # Model Training Script
│   ├── predict.py             # Stock Price Prediction Script
├── requirements.txt      # Dependencies
├── README.md             # Project Documentation
```

## 🔮 Future Improvements
- Include additional financial indicators (RSI, MACD, Moving Averages)
- Implement GRU models for comparison
- Develop a Web App for real-time predictions

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 💡 Acknowledgments
- Yahoo Finance for stock data
- TensorFlow & Keras for Deep Learning

⭐ If you find this project useful, consider giving it a star! ⭐
