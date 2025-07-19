import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    MODELS_AVAILABLE = True
except ImportError:
    st.error("Please install required packages: pip install tensorflow scikit-learn")
    MODELS_AVAILABLE = False

import ta
from datetime import datetime, timedelta
import time

# Configuration
st.set_page_config(
    page_title="Crypto Price Prediction System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CryptoDataFetcher:
    def __init__(self):
        self.crypto_symbols = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Binance Coin': 'BNB-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'Polkadot': 'DOT-USD',
            'Dogecoin': 'DOGE-USD',
            'Avalanche': 'AVAX-USD',
            'Polygon': 'MATIC-USD',
            'Chainlink': 'LINK-USD'
        }
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_crypto_data(_self, symbol, period='2y'):
        """Fetch cryptocurrency data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_fundamental_data(_self, crypto_id):
        """Fetch fundamental data from CoinGecko API"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.warning(f"Could not fetch fundamental data: {str(e)}")
            return None

class TechnicalAnalyzer:
    @staticmethod
    def add_technical_indicators(df):
        """Add comprehensive technical indicators"""
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Stochastic
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df
    
    @staticmethod
    def generate_signals(df):
        """Generate trading signals based on technical indicators"""
        signals = []
        
        # MACD Signal
        if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            signals.append("🟢 MACD: Bullish")
        else:
            signals.append("🔴 MACD: Bearish")
        
        # RSI Signal
        rsi_current = df['RSI'].iloc[-1]
        if rsi_current > 70:
            signals.append("🔴 RSI: Overbought")
        elif rsi_current < 30:
            signals.append("🟢 RSI: Oversold")
        else:
            signals.append("🟡 RSI: Neutral")
        
        # Moving Average Signal
        if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
            signals.append("🟢 MA: Above 50-day SMA")
        else:
            signals.append("🔴 MA: Below 50-day SMA")
        
        # Bollinger Bands
        if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
            signals.append("🔴 BB: Above Upper Band")
        elif df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
            signals.append("🟢 BB: Below Lower Band")
        else:
            signals.append("🟡 BB: Within Bands")
        
        return signals

class ModelTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
        
    def prepare_data(self, df, lookback=60):
        """Prepare data for machine learning models"""
        # Select features for training
        features = ['Close', 'Volume', 'High', 'Low', 'SMA_20', 'SMA_50', 
                   'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'OBV']
        
        # Remove rows with NaN values
        df_clean = df[features].dropna()
        
        if len(df_clean) < lookback + 10:
            raise ValueError("Insufficient data for training")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df_clean)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y), df_clean
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        return model
    
    def train_models(self, X, y):
        """Train multiple models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models_performance = {}
        
        # LSTM Model
        with st.spinner("Training LSTM model..."):
            lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            lstm_model.fit(X_train, y_train, 
                          epochs=50, 
                          batch_size=32, 
                          validation_split=0.2, 
                          callbacks=[early_stopping], 
                          verbose=0)
            
            lstm_pred = lstm_model.predict(X_test, verbose=0)
            models_performance['LSTM'] = {
                'model': lstm_model,
                'mse': mean_squared_error(y_test, lstm_pred),
                'mae': mean_absolute_error(y_test, lstm_pred),
                'r2': r2_score(y_test, lstm_pred)
            }
        
        # GRU Model
        with st.spinner("Training GRU model..."):
            gru_model = self.build_gru_model((X_train.shape[1], X_train.shape[2]))
            gru_model.fit(X_train, y_train, 
                         epochs=50, 
                         batch_size=32, 
                         validation_split=0.2, 
                         callbacks=[early_stopping], 
                         verbose=0)
            
            gru_pred = gru_model.predict(X_test, verbose=0)
            models_performance['GRU'] = {
                'model': gru_model,
                'mse': mean_squared_error(y_test, gru_pred),
                'mae': mean_absolute_error(y_test, gru_pred),
                'r2': r2_score(y_test, gru_pred)
            }
        
        # Traditional ML Models (using last timestep features)
        X_traditional = X[:, -1, :]  # Use last timestep
        X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(
            X_traditional, y, test_size=0.2, random_state=42)
        
        # Random Forest
        with st.spinner("Training Random Forest model..."):
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_trad, y_train_trad)
            rf_pred = rf_model.predict(X_test_trad)
            
            models_performance['Random Forest'] = {
                'model': rf_model,
                'mse': mean_squared_error(y_test_trad, rf_pred),
                'mae': mean_absolute_error(y_test_trad, rf_pred),
                'r2': r2_score(y_test_trad, rf_pred)
            }
        
        # Gradient Boosting
        with st.spinner("Training Gradient Boosting model..."):
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train_trad, y_train_trad)
            gb_pred = gb_model.predict(X_test_trad)
            
            models_performance['Gradient Boosting'] = {
                'model': gb_model,
                'mse': mean_squared_error(y_test_trad, gb_pred),
                'mae': mean_absolute_error(y_test_trad, gb_pred),
                'r2': r2_score(y_test_trad, gb_pred)
            }
        
        return models_performance, X_test, y_test
    
    def make_predictions(self, model, model_type, data, days=30):
        """Make future predictions"""
        predictions = []
        current_sequence = data[-60:].copy()  # Use last 60 days
        
        for _ in range(days):
            if model_type in ['LSTM', 'GRU']:
                pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
                pred_value = pred[0, 0]
            else:  # Traditional ML models
                pred_value = model.predict(current_sequence[-1:].reshape(1, -1))[0]
            
            predictions.append(pred_value)
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_value  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🚀 Crypto Price Prediction System
        </h1>
        <p style="color: white; text-align: center; margin: 0;">
            Advanced ML-based cryptocurrency analysis with fundamental & technical indicators
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODELS_AVAILABLE:
        st.stop()
    
    # Initialize classes
    fetcher = CryptoDataFetcher()
    analyzer = TechnicalAnalyzer()
    trainer = ModelTrainer()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Crypto selection
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(fetcher.crypto_symbols.keys())
    )
    
    symbol = fetcher.crypto_symbols[selected_crypto]
    
    # Time period
    time_period = st.sidebar.selectbox(
        "Historical Data Period",
        options=['1y', '2y', '5y', 'max'],
        index=1
    )
    
    # Prediction days
    pred_days = st.sidebar.slider(
        "Prediction Days",
        min_value=7,
        max_value=90,
        value=30
    )
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models to Train",
        options=['LSTM', 'GRU', 'Random Forest', 'Gradient Boosting'],
        default=['LSTM', 'Random Forest']
    )
    
    # Fetch and process data
    if st.sidebar.button("Start Analysis", type="primary"):
        with st.spinner("Fetching cryptocurrency data..."):
            data = fetcher.get_crypto_data(symbol, time_period)
            
            if data is not None and len(data) > 100:
                # Add technical indicators
                data_with_indicators = analyzer.add_technical_indicators(data)
                
                # Display current price and basic info
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.2f}",
                        delta=f"{price_change_pct:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="24h High",
                        value=f"${data['High'].iloc[-1]:.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="24h Low",
                        value=f"${data['Low'].iloc[-1]:.2f}"
                    )
                
                with col4:
                    st.metric(
                        label="Volume",
                        value=f"{data['Volume'].iloc[-1]:,.0f}"
                    )
                
                # Price chart with technical indicators
                st.subheader("Price Chart with Technical Analysis")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD')
                )
                
                # Price and moving averages
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data_with_indicators['SMA_20'],
                    mode='lines', name='SMA 20', line=dict(color='orange')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data_with_indicators['SMA_50'],
                    mode='lines', name='SMA 50', line=dict(color='blue')
                ), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=data.index, y=data_with_indicators['RSI'],
                    mode='lines', name='RSI', line=dict(color='purple')
                ), row=2, col=1)
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=data.index, y=data_with_indicators['MACD'],
                    mode='lines', name='MACD', line=dict(color='blue')
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=data_with_indicators['MACD_signal'],
                    mode='lines', name='Signal', line=dict(color='red')
                ), row=3, col=1)
                
                fig.update_layout(
                    title=f"{selected_crypto} Technical Analysis",
                    xaxis_title="Date",
                    height=800,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical signals
                st.subheader("Technical Analysis Signals")
                signals = analyzer.generate_signals(data_with_indicators)
                
                cols = st.columns(2)
                for i, signal in enumerate(signals):
                    with cols[i % 2]:
                        st.write(signal)
                
                # Model Training and Predictions
                st.subheader("Machine Learning Predictions")
                
                try:
                    X, y, df_clean = trainer.prepare_data(data_with_indicators)
                    
                    if len(selected_models) > 0:
                        models_performance, X_test, y_test = trainer.train_models(X, y)
                        
                        # Filter models based on selection
                        selected_performance = {k: v for k, v in models_performance.items() 
                                             if k in selected_models}
                        
                        # Display model performance
                        st.subheader("Model Performance Comparison")
                        
                        performance_df = pd.DataFrame({
                            'Model': list(selected_performance.keys()),
                            'MSE': [v['mse'] for v in selected_performance.values()],
                            'MAE': [v['mae'] for v in selected_performance.values()],
                            'R²': [v['r2'] for v in selected_performance.values()]
                        })
                        
                        st.dataframe(performance_df.round(4))
                        
                        # Select best model
                        best_model_name = min(selected_performance.keys(), 
                                            key=lambda x: selected_performance[x]['mse'])
                        best_model = selected_performance[best_model_name]['model']
                        
                        st.success(f"Best performing model: {best_model_name}")
                        
                        # Make predictions
                        st.subheader(f"Price Predictions - Next {pred_days} Days")
                        
                        scaled_data = trainer.scaler.transform(df_clean)
                        predictions = trainer.make_predictions(
                            best_model, best_model_name, scaled_data, pred_days
                        )
                        
                        # Inverse transform predictions
                        pred_array = np.zeros((len(predictions), df_clean.shape[1]))
                        pred_array[:, 0] = predictions
                        predictions_original = trainer.scaler.inverse_transform(pred_array)[:, 0]
                        
                        # Create future dates
                        last_date = data.index[-1]
                        future_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=pred_days,
                            freq='D'
                        )
                        
                        # Prediction chart
                        fig_pred = go.Figure()
                        
                        # Historical data
                        fig_pred.add_trace(go.Scatter(
                            x=data.index[-90:],  # Last 90 days
                            y=data['Close'].iloc[-90:],
                            mode='lines',
                            name='Historical Price',
                            line=dict(color='blue')
                        ))
                        
                        # Predictions
                        fig_pred.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions_original,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_pred.update_layout(
                            title=f"{selected_crypto} Price Prediction using {best_model_name}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Prediction summary
                        current_price = data['Close'].iloc[-1]
                        predicted_price = predictions_original[-1]
                        price_change_pred = ((predicted_price - current_price) / current_price) * 100
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Prediction Summary</h3>
                            <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                            <p><strong>Predicted Price ({pred_days} days):</strong> ${predicted_price:.2f}</p>
                            <p><strong>Expected Change:</strong> {price_change_pred:+.2f}%</p>
                            <p><strong>Model Used:</strong> {best_model_name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk Assessment
                        st.subheader("Risk Assessment")
                        
                        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                        
                        if volatility > 100:
                            risk_level = "🔴 Very High"
                        elif volatility > 75:
                            risk_level = "🟠 High"
                        elif volatility > 50:
                            risk_level = "🟡 Medium"
                        else:
                            risk_level = "🟢 Low"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Annualized Volatility", f"{volatility:.1f}%")
                        with col2:
                            st.metric("Risk Level", risk_level)
                        
                        # Download predictions
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Price': predictions_original
                        })
                        
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_crypto}_predictions.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error in model training: {str(e)}")
                    st.info("This might be due to insufficient data or technical indicator calculation issues.")
            
            else:
                st.error("Unable to fetch sufficient data for analysis.")
    
    # About section
    with st.expander("About this Application"):
        st.markdown("""
        ### Features:
        - **Real-time Data**: Fetches live cryptocurrency data from Yahoo Finance
        - **Technical Analysis**: 15+ technical indicators including RSI, MACD, Bollinger Bands
        - **Multiple ML Models**: LSTM, GRU, Random Forest, Gradient Boosting
        - **Fundamental Analysis**: Market cap, volume, and other key metrics
        - **Risk Assessment**: Volatility analysis and risk scoring
        - **Interactive Charts**: Plotly-based interactive visualizations
        - **Predictions Export**: Download predictions as CSV
        
        ### Models Used:
        - **LSTM**: Long Short-Term Memory networks for sequence prediction
        - **GRU**: Gated Recurrent Units for time series forecasting
        - **Random Forest**: Ensemble method for robust predictions
        - **Gradient Boosting**: Advanced boosting technique
        
        ### Disclaimer:
        This tool is for educational purposes only. Cryptocurrency investments carry high risk.
        Always do your own research before making investment decisions.
        """)

if __name__ == "__main__":
    main()
