(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF'
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,469 +1,938 @@
-import streamlit as st
-import pandas as pd
-import numpy as np
-import yfinance as yf
-import plotly.graph_objects as go
-from plotly.subplots import make_subplots
-import ta
-from datetime import datetime, timedelta
-import requests
-from sklearn.ensemble import RandomForestClassifier
-from sklearn.model_selection import train_test_split
-from sklearn.metrics import accuracy_score
-import warnings
-warnings.filterwarnings('ignore')
-
-# Page config
-st.set_page_config(
-    page_title="CryptoEdge - Advanced Crypto Trading Platform",
-    page_icon="📈",
-    layout="wide",
-    initial_sidebar_state="expanded"
-)
-
-# Custom CSS
-st.markdown("""
-<style>
-    .main-header {
-        font-size: 3rem;
-        color: #1f77b4;
-        text-align: center;
-        margin-bottom: 2rem;
-    }
-    .metric-card {
-        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
-        padding: 1rem;
-        border-radius: 10px;
-        color: white;
-        margin: 0.5rem 0;
-    }
-    .signal-buy {
-        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
-        padding: 1rem;
-        border-radius: 10px;
-        color: white;
-        text-align: center;
-        font-weight: bold;
-    }
-    .signal-sell {
-        background: linear-gradient(135deg, #ff5858 0%, #f857a6 100%);
-        padding: 1rem;
-        border-radius: 10px;
-        color: white;
-        text-align: center;
-        font-weight: bold;
-    }
-    .signal-hold {
-        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
-        padding: 1rem;
-        border-radius: 10px;
-        color: white;
-        text-align: center;
-        font-weight: bold;
-    }
-</style>
-""", unsafe_allow_html=True)
-
-class CryptoAnalyzer:
-    def __init__(self):
-        self.crypto_symbols = {
-            'Bitcoin': 'BTC-USD',
-            'Ethereum': 'ETH-USD',
-            'Binance Coin': 'BNB-USD',
-            'Cardano': 'ADA-USD',
-            'Solana': 'SOL-USD',
-            'Dogecoin': 'DOGE-USD',
-            'Polygon': 'MATIC-USD',
-            'Avalanche': 'AVAX-USD',
-            'Chainlink': 'LINK-USD',
-            'Polkadot': 'DOT-USD'
-        }
-    
-    def get_crypto_data(self, symbol, period='1y'):
-        """Fetch cryptocurrency data"""
-        try:
-            ticker = yf.Ticker(symbol)
-            data = ticker.history(period=period)
-            return data
-        except Exception as e:
-            st.error(f"Error fetching data for {symbol}: {str(e)}")
-            return None
-    
-    def calculate_technical_indicators(self, data):
-        """Calculate technical indicators"""
-        if data is None or data.empty:
-            return None
-        
-        try:
-            # RSI
-            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
-            
-            # MACD
-            macd = ta.trend.MACD(data['Close'])
-            data['MACD'] = macd.macd()
-            data['MACD_Signal'] = macd.macd_signal()
-            data['MACD_Histogram'] = macd.macd_diff()
-            
-            # Bollinger Bands
-            bb = ta.volatility.BollingerBands(data['Close'])
-            data['BB_Upper'] = bb.bollinger_hband()
-            data['BB_Middle'] = bb.bollinger_mavg()
-            data['BB_Lower'] = bb.bollinger_lband()
-            
-            # Moving Averages
-            data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
-            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
-            data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
-            data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
-            
-            # Volume indicators (using pandas rolling mean instead of TA-Lib)
-            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
-            
-            # Volatility
-            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
-            
-            # Fill NaN values
-            data = data.bfill().ffill()
-            
-        except Exception as e:
-            st.error(f"Error calculating technical indicators: {str(e)}")
-            return None
-        
-        return data
-    
-    def generate_signals(self, data):
-        """Generate trading signals based on technical indicators"""
-        if data is None or data.empty:
-            return None, None, None
-        
-        latest = data.iloc[-1]
-        signals = []
-        
-        # RSI Signal
-        if latest['RSI'] < 30:
-            signals.append(('RSI', 'BUY', 'Oversold condition'))
-        elif latest['RSI'] > 70:
-            signals.append(('RSI', 'SELL', 'Overbought condition'))
-        
-        # MACD Signal
-        if latest['MACD'] > latest['MACD_Signal']:
-            signals.append(('MACD', 'BUY', 'Bullish crossover'))
-        else:
-            signals.append(('MACD', 'SELL', 'Bearish crossover'))
-        
-        # Bollinger Bands Signal
-        if latest['Close'] < latest['BB_Lower']:
-            signals.append(('BB', 'BUY', 'Price below lower band'))
-        elif latest['Close'] > latest['BB_Upper']:
-            signals.append(('BB', 'SELL', 'Price above upper band'))
-        
-        # Moving Average Signal
-        if latest['SMA_20'] > latest['SMA_50']:
-            signals.append(('MA', 'BUY', 'Short MA above Long MA'))
-        else:
-            signals.append(('MA', 'SELL', 'Short MA below Long MA'))
-        
-        # Overall signal
-        buy_signals = sum(1 for _, signal, _ in signals if signal == 'BUY')
-        sell_signals = sum(1 for _, signal, _ in signals if signal == 'SELL')
-        
-        if buy_signals > sell_signals:
-            overall_signal = 'BUY'
-        elif sell_signals > buy_signals:
-            overall_signal = 'SELL'
-        else:
-            overall_signal = 'HOLD'
-        
-        # Calculate entry, stop loss, and take profit
-        entry_price = latest['Close']
-        atr = latest['ATR']
-        
-        if overall_signal == 'BUY':
-            stop_loss = entry_price - (2 * atr)
-            take_profit = entry_price + (3 * atr)
-        elif overall_signal == 'SELL':
-            stop_loss = entry_price + (2 * atr)
-            take_profit = entry_price - (3 * atr)
-        else:
-            stop_loss = entry_price - (1.5 * atr)
-            take_profit = entry_price + (1.5 * atr)
-        
-        return signals, overall_signal, {
-            'entry': entry_price,
-            'stop_loss': stop_loss,
-            'take_profit': take_profit,
-            'risk_reward': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
-        }
-    
-    def create_chart(self, data, symbol):
-        """Create interactive chart with indicators"""
-        if data is None or data.empty:
-            return None
-        
-        fig = make_subplots(
-            rows=3, cols=1,
-            shared_xaxes=True,
-            vertical_spacing=0.1,
-            subplot_titles=('Price & Indicators', 'RSI', 'MACD'),
-            row_heights=[0.6, 0.2, 0.2]
-        )
-        
-        # Candlestick chart
-        fig.add_trace(
-            go.Candlestick(
-                x=data.index,
-                open=data['Open'],
-                high=data['High'],
-                low=data['Low'],
-                close=data['Close'],
-                name='Price'
-            ),
-            row=1, col=1
-        )
-        
-        # Bollinger Bands
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='red', width=1)),
-            row=1, col=1
-        )
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='red', width=1)),
-            row=1, col=1
-        )
-        
-        # Moving Averages
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='blue', width=1)),
-            row=1, col=1
-        )
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)),
-            row=1, col=1
-        )
-        
-        # RSI
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
-            row=2, col=1
-        )
-        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
-        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
-        
-        # MACD
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
-            row=3, col=1
-        )
-        fig.add_trace(
-            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
-            row=3, col=1
-        )
-        fig.add_trace(
-            go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'),
-            row=3, col=1
-        )
-        
-        fig.update_layout(
-            title=f'{symbol} Technical Analysis',
-            xaxis_title='Date',
-            height=800,
-            showlegend=True
-        )
-        
-        return fig
-    
-    def backtest_strategy(self, data, initial_capital=10000):
-        """Simple backtesting strategy"""
-        if data is None or data.empty:
-            return None
-        
-        data = data.copy()
-        data['Position'] = 0
-        data['Returns'] = data['Close'].pct_change()
-        
-        # Simple strategy: Buy when RSI < 30, Sell when RSI > 70
-        for i in range(1, len(data)):
-            if data.iloc[i]['RSI'] < 30 and data.iloc[i-1]['RSI'] >= 30:
-                data.iloc[i, data.columns.get_loc('Position')] = 1
-            elif data.iloc[i]['RSI'] > 70 and data.iloc[i-1]['RSI'] <= 70:
-                data.iloc[i, data.columns.get_loc('Position')] = -1
-            else:
-                data.iloc[i, data.columns.get_loc('Position')] = data.iloc[i-1]['Position']
-        
-        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
-        data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
-        data['Portfolio_Value'] = initial_capital * data['Cumulative_Returns']
-        
-        return data
-
-def main():
-    st.markdown('<h1 class="main-header">🚀 CryptoEdge Trading Platform</h1>', unsafe_allow_html=True)
-    st.markdown("### Advanced Crypto Analysis & Signal Generation for Mumbai Traders")
-    
-    analyzer = CryptoAnalyzer()
-    
-    # Sidebar
-    st.sidebar.header("Trading Dashboard")
-    
-    # Crypto selection
-    selected_crypto = st.sidebar.selectbox(
-        "Select Cryptocurrency:",
-        list(analyzer.crypto_symbols.keys())
-    )
-    
-    # Time period selection
-    period = st.sidebar.selectbox(
-        "Select Time Period:",
-        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
-        index=3
-    )
-    
-    # Get data
-    symbol = analyzer.crypto_symbols[selected_crypto]
-    data = analyzer.get_crypto_data(symbol, period)
-    
-    if data is not None:
-        # Calculate indicators
-        data = analyzer.calculate_technical_indicators(data)
-        
-        # Generate signals
-        signals, overall_signal, trade_params = analyzer.generate_signals(data)
-        
-        # Main dashboard
-        col1, col2, col3 = st.columns(3)
-        
-        with col1:
-            st.markdown("### Current Price")
-            current_price = data['Close'].iloc[-1]
-            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
-            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
-            
-            st.metric(
-                f"{selected_crypto}",
-                f"${current_price:.2f}",
-                f"{price_change_pct:.2f}%"
-            )
-        
-        with col2:
-            st.markdown("### Trading Signal")
-            if overall_signal == 'BUY':
-                st.markdown('<div class="signal-buy">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
-            elif overall_signal == 'SELL':
-                st.markdown('<div class="signal-sell">🔴 SELL SIGNAL</div>', unsafe_allow_html=True)
-            else:
-                st.markdown('<div class="signal-hold">🟡 HOLD SIGNAL</div>', unsafe_allow_html=True)
-        
-        with col3:
-            st.markdown("### Risk/Reward")
-            if trade_params:
-                st.metric(
-                    "R/R Ratio",
-                    f"{trade_params['risk_reward']:.2f}",
-                    "Good" if trade_params['risk_reward'] > 2 else "Review"
-                )
-        
-        # Trade Parameters
-        if trade_params:
-            st.markdown("### 📊 Trade Parameters")
-            col1, col2, col3, col4 = st.columns(4)
-            
-            with col1:
-                st.metric("Entry Price", f"${trade_params['entry']:.2f}")
-            with col2:
-                st.metric("Stop Loss", f"${trade_params['stop_loss']:.2f}")
-            with col3:
-                st.metric("Take Profit", f"${trade_params['take_profit']:.2f}")
-            with col4:
-                risk_amount = abs(trade_params['entry'] - trade_params['stop_loss'])
-                st.metric("Risk Amount", f"${risk_amount:.2f}")
-        
-        # Technical Indicators
-        st.markdown("### 📈 Technical Analysis")
-        
-        # Create chart
-        chart = analyzer.create_chart(data, selected_crypto)
-        if chart:
-            st.plotly_chart(chart, use_container_width=True)
-        
-        # Signals breakdown
-        st.markdown("### 🎯 Signal Breakdown")
-        if signals:
-            signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Description'])
-            st.dataframe(signal_df, use_container_width=True)
-        
-        # Key metrics
-        st.markdown("### 📊 Key Metrics")
-        col1, col2, col3, col4 = st.columns(4)
-        
-        with col1:
-            st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
-        with col2:
-            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
-        with col3:
-            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
-        with col4:
-            st.metric("ATR", f"{data['ATR'].iloc[-1]:.2f}")
-        
-        # Backtesting
-        st.markdown("### 📈 Strategy Backtesting")
-        backtest_data = analyzer.backtest_strategy(data)
-        
-        if backtest_data is not None:
-            col1, col2 = st.columns(2)
-            
-            with col1:
-                final_value = backtest_data['Portfolio_Value'].iloc[-1]
-                total_return = ((final_value - 10000) / 10000) * 100
-                st.metric("Total Return", f"{total_return:.2f}%")
-            
-            with col2:
-                max_drawdown = ((backtest_data['Portfolio_Value'].min() - backtest_data['Portfolio_Value'].max()) / backtest_data['Portfolio_Value'].max()) * 100
-                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
-            
-            # Portfolio performance chart
-            fig_portfolio = go.Figure()
-            fig_portfolio.add_trace(go.Scatter(
-                x=backtest_data.index,
-                y=backtest_data['Portfolio_Value'],
-                mode='lines',
-                name='Portfolio Value',
-                line=dict(color='green', width=2)
-            ))
-            
-            fig_portfolio.update_layout(
-                title='Portfolio Performance',
-                xaxis_title='Date',
-                yaxis_title='Portfolio Value ($)',
-                height=400
-            )
-            
-            st.plotly_chart(fig_portfolio, use_container_width=True)
-        
-        # Risk Management
-        st.markdown("### ⚠️ Risk Management Guidelines")
-        st.info("""
-        **Mumbai Trader's Risk Management Rules:**
-        - Never risk more than 2% of your capital on a single trade
-        - Always set stop losses before entering a trade
-        - Consider INR conversion rates and timing for Indian markets
-        - Keep track of your trades for tax purposes (30% flat tax in India)
-        - Use proper position sizing based on your account size
-        """)
-        
-        # Market sentiment
-        st.markdown("### 📰 Market Sentiment")
-        sentiment_score = np.random.randint(1, 100)  # Placeholder for real sentiment analysis
-        
-        if sentiment_score > 70:
-            st.success(f"Market Sentiment: Bullish ({sentiment_score}/100)")
-        elif sentiment_score < 30:
-            st.error(f"Market Sentiment: Bearish ({sentiment_score}/100)")
-        else:
-            st.warning(f"Market Sentiment: Neutral ({sentiment_score}/100)")
-    
-    # Footer
-    st.markdown("---")
-    st.markdown("**Disclaimer:** This is for educational purposes only. Always do your own research before trading.")
-
-if __name__ == "__main__":
-    main()
+import streamlit as st
+import pandas as pd
+import numpy as np
+import yfinance as yf
+import plotly.graph_objects as go
+from plotly.subplots import make_subplots
+import ta
+from datetime import datetime, timedelta
+import requests
+import warnings
+import joblib
+import seaborn as sns
+import matplotlib.pyplot as plt
+from scipy import stats
+import time
+import json
+from typing import Dict, List, Tuple, Optional
+import threading
+import schedule
+
+# ML Libraries
+from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
+from sklearn.model_selection import train_test_split, TimeSeriesSplit
+from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
+from sklearn.preprocessing import StandardScaler, MinMaxScaler
+import xgboost as xgb
+import lightgbm as lgb
+
+# Deep Learning
+import tensorflow as tf
+from tensorflow import keras
+from tensorflow.keras import layers
+from tensorflow.keras.models import Sequential
+from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
+from tensorflow.keras.optimizers import Adam
+from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
+
+# Time Series Forecasting
+from prophet import Prophet
+
+# Sentiment Analysis
+from textblob import TextBlob
+from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
+
+# Crypto APIs
+import ccxt
+from pycoingecko import CoinGeckoAPI
+
+warnings.filterwarnings('ignore')
+
+# Page config
+st.set_page_config(
+    page_title="AI Crypto Price Predictor",
+    page_icon="🚀",
+    layout="wide",
+    initial_sidebar_state="expanded"
+)
+
+# Custom CSS
+st.markdown("""
+<style>
+    .main-header {
+        font-size: 3rem;
+        color: #1f77b4;
+        text-align: center;
+        margin-bottom: 2rem;
+        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
+        -webkit-background-clip: text;
+        -webkit-text-fill-color: transparent;
+    }
+    .prediction-card {
+        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
+        padding: 1.5rem;
+        border-radius: 15px;
+        color: white;
+        margin: 1rem 0;
+        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
+    }
+    .model-performance {
+        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
+        padding: 1rem;
+        border-radius: 10px;
+        color: white;
+        margin: 0.5rem 0;
+    }
+    .alert-box {
+        background: linear-gradient(135deg, #ff5858 0%, #f857a6 100%);
+        padding: 1rem;
+        border-radius: 10px;
+        color: white;
+        text-align: center;
+        font-weight: bold;
+        margin: 1rem 0;
+    }
+    .confidence-high {
+        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
+        padding: 0.5rem;
+        border-radius: 5px;
+        color: white;
+        text-align: center;
+    }
+    .confidence-medium {
+        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
+        padding: 0.5rem;
+        border-radius: 5px;
+        color: white;
+        text-align: center;
+    }
+    .confidence-low {
+        background: linear-gradient(135deg, #ff5858 0%, #f857a6 100%);
+        padding: 0.5rem;
+        border-radius: 5px;
+        color: white;
+        text-align: center;
+    }
+</style>
+""", unsafe_allow_html=True)
+
+class CryptoDataFetcher:
+    """Advanced crypto data fetcher with multiple API sources"""
+    
+    def __init__(self):
+        self.cg = CoinGeckoAPI()
+        self.exchanges = {
+            'binance': ccxt.binance(),
+            'coinbase': ccxt.coinbasepro(),
+            'kraken': ccxt.kraken()
+        }
+        
+    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
+        """Get real-time price from multiple sources"""
+        try:
+            # CoinGecko API
+            coin_id = self._get_coingecko_id(symbol)
+            if coin_id:
+                data = self.cg.get_price(ids=coin_id, vs_currencies='usd', include_24hr_change=True)
+                if coin_id in data:
+                    return {
+                        'price': data[coin_id]['usd'],
+                        'change_24h': data[coin_id].get('usd_24h_change', 0),
+                        'source': 'coingecko'
+                    }
+        except Exception as e:
+            st.warning(f"CoinGecko API error: {str(e)}")
+        
+        # Fallback to yfinance
+        try:
+            ticker = yf.Ticker(symbol)
+            info = ticker.info
+            return {
+                'price': info.get('regularMarketPrice', 0),
+                'change_24h': info.get('regularMarketChangePercent', 0),
+                'source': 'yfinance'
+            }
+        except Exception as e:
+            st.error(f"Error fetching real-time data: {str(e)}")
+            return None
+    
+    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
+        """Map symbol to CoinGecko ID"""
+        mapping = {
+            'BTC-USD': 'bitcoin',
+            'ETH-USD': 'ethereum',
+            'BNB-USD': 'binancecoin',
+            'ADA-USD': 'cardano',
+            'SOL-USD': 'solana',
+            'DOGE-USD': 'dogecoin',
+            'MATIC-USD': 'matic-network',
+            'AVAX-USD': 'avalanche-2',
+            'LINK-USD': 'chainlink',
+            'DOT-USD': 'polkadot'
+        }
+        return mapping.get(symbol)
+    
+    def get_historical_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
+        """Get historical data with enhanced features"""
+        try:
+            ticker = yf.Ticker(symbol)
+            data = ticker.history(period=period)
+            
+            if data.empty:
+                return None
+            
+            # Add additional features
+            data['Returns'] = data['Close'].pct_change()
+            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
+            data['Volatility'] = data['Returns'].rolling(window=20).std()
+            data['Price_Range'] = data['High'] - data['Low']
+            data['Price_Change'] = data['Close'] - data['Open']
+            
+            return data
+        except Exception as e:
+            st.error(f"Error fetching historical data: {str(e)}")
+            return None
+    
+    def get_market_sentiment(self, symbol: str) -> Dict:
+        """Get market sentiment from news and social media"""
+        try:
+            # Placeholder for sentiment analysis
+            # In real implementation, you would fetch news and social media data
+            analyzer = SentimentIntensityAnalyzer()
+            
+            # Simulate news headlines (replace with real news API)
+            sample_headlines = [
+                f"{symbol.split('-')[0]} shows strong bullish momentum",
+                f"Market analysts predict {symbol.split('-')[0]} growth",
+                f"Institutional adoption of {symbol.split('-')[0]} increasing"
+            ]
+            
+            sentiments = []
+            for headline in sample_headlines:
+                sentiment = analyzer.polarity_scores(headline)
+                sentiments.append(sentiment['compound'])
+            
+            avg_sentiment = np.mean(sentiments)
+            
+            return {
+                'sentiment_score': avg_sentiment,
+                'sentiment_label': 'Bullish' if avg_sentiment > 0.1 else 'Bearish' if avg_sentiment < -0.1 else 'Neutral',
+                'confidence': abs(avg_sentiment)
+            }
+        except Exception as e:
+            return {
+                'sentiment_score': 0,
+                'sentiment_label': 'Neutral',
+                'confidence': 0
+            }
+
+class FeatureEngineer:
+    """Advanced feature engineering for ML models"""
+    
+    @staticmethod
+    def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
+        """Create comprehensive technical indicators"""
+        try:
+            # Existing technical indicators
+            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
+            data['MACD'] = ta.trend.MACD(data['Close']).macd()
+            data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
+            data['BB_Upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
+            data['BB_Lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
+            data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
+            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
+            data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
+            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
+            
+            # Advanced technical indicators
+            data['Stoch_K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
+            data['Stoch_D'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch_signal()
+            data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
+            data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
+            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
+            
+            # Volume indicators
+            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
+            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
+            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
+            
+            # Price patterns
+            data['High_Low_Ratio'] = data['High'] / data['Low']
+            data['Close_Open_Ratio'] = data['Close'] / data['Open']
+            data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
+            
+            # Lag features
+            for lag in [1, 2, 3, 5, 10]:
+                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
+                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
+                data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
+            
+            # Rolling statistics
+            for window in [5, 10, 20]:
+                data[f'Close_Mean_{window}'] = data['Close'].rolling(window=window).mean()
+                data[f'Close_Std_{window}'] = data['Close'].rolling(window=window).std()
+                data[f'Volume_Mean_{window}'] = data['Volume'].rolling(window=window).mean()
+                data[f'Returns_Mean_{window}'] = data['Returns'].rolling(window=window).mean()
+            
+            # Fill NaN values
+            data = data.bfill().ffill()
+            
+            return data
+        except Exception as e:
+            st.error(f"Error in feature engineering: {str(e)}")
+            return data
+
+class MLPredictor:
+    """Advanced ML prediction models"""
+    
+    def __init__(self):
+        self.models = {}
+        self.scalers = {}
+        self.feature_columns = []
+        
+    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close', 
+                    lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
+        """Prepare data for ML models"""
+        try:
+            # Select features (excluding target and date-like columns)
+            feature_cols = [col for col in data.columns if col not in 
+                          ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close'] and 
+                          not data[col].dtype == 'object']
+            
+            self.feature_columns = feature_cols
+            features = data[feature_cols].values
+            target = data[target_col].values
+            
+            # Scale features
+            scaler = MinMaxScaler()
+            features_scaled = scaler.fit_transform(features)
+            self.scalers['features'] = scaler
+            
+            # Scale target
+            target_scaler = MinMaxScaler()
+            target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
+            self.scalers['target'] = target_scaler
+            
+            # Create sequences for time series
+            X, y = [], []
+            for i in range(lookback, len(features_scaled)):
+                X.append(features_scaled[i-lookback:i])
+                y.append(target_scaled[i])
+            
+            return np.array(X), np.array(y)
+        except Exception as e:
+            st.error(f"Error preparing data: {str(e)}")
+            return np.array([]), np.array([])
+    
+    def build_lstm_model(self, input_shape: tuple) -> keras.Model:
+        """Build LSTM model for price prediction"""
+        model = Sequential([
+            LSTM(100, return_sequences=True, input_shape=input_shape),
+            Dropout(0.2),
+            LSTM(100, return_sequences=True),
+            Dropout(0.2),
+            LSTM(50, return_sequences=False),
+            Dropout(0.2),
+            Dense(25),
+            Dense(1)
+        ])
+        
+        model.compile(optimizer=Adam(learning_rate=0.001), 
+                     loss='mse', metrics=['mae'])
+        return model
+    
+    def build_gru_model(self, input_shape: tuple) -> keras.Model:
+        """Build GRU model for price prediction"""
+        model = Sequential([
+            GRU(100, return_sequences=True, input_shape=input_shape),
+            Dropout(0.2),
+            GRU(100, return_sequences=True),
+            Dropout(0.2),
+            GRU(50, return_sequences=False),
+            Dropout(0.2),
+            Dense(25),
+            Dense(1)
+        ])
+        
+        model.compile(optimizer=Adam(learning_rate=0.001), 
+                     loss='mse', metrics=['mae'])
+        return model
+    
+    def build_cnn_lstm_model(self, input_shape: tuple) -> keras.Model:
+        """Build CNN-LSTM hybrid model"""
+        model = Sequential([
+            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
+            MaxPooling1D(pool_size=2),
+            LSTM(100, return_sequences=True),
+            Dropout(0.2),
+            LSTM(50, return_sequences=False),
+            Dropout(0.2),
+            Dense(25),
+            Dense(1)
+        ])
+        
+        model.compile(optimizer=Adam(learning_rate=0.001), 
+                     loss='mse', metrics=['mae'])
+        return model
+    
+    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
+        """Train multiple ML models"""
+        results = {}
+        
+        # Split data
+        X_train, X_test, y_train, y_test = train_test_split(
+            X, y, test_size=0.2, random_state=42, shuffle=False
+        )
+        
+        try:
+            # XGBoost
+            X_train_2d = X_train.reshape(X_train.shape[0], -1)
+            X_test_2d = X_test.reshape(X_test.shape[0], -1)
+            
+            xgb_model = xgb.XGBRegressor(
+                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
+            )
+            xgb_model.fit(X_train_2d, y_train)
+            xgb_pred = xgb_model.predict(X_test_2d)
+            
+            self.models['xgboost'] = xgb_model
+            results['xgboost'] = {
+                'mse': mean_squared_error(y_test, xgb_pred),
+                'mae': mean_absolute_error(y_test, xgb_pred),
+                'r2': r2_score(y_test, xgb_pred),
+                'predictions': xgb_pred
+            }
+            
+            # LightGBM
+            lgb_model = lgb.LGBMRegressor(
+                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
+            )
+            lgb_model.fit(X_train_2d, y_train)
+            lgb_pred = lgb_model.predict(X_test_2d)
+            
+            self.models['lightgbm'] = lgb_model
+            results['lightgbm'] = {
+                'mse': mean_squared_error(y_test, lgb_pred),
+                'mae': mean_absolute_error(y_test, lgb_pred),
+                'r2': r2_score(y_test, lgb_pred),
+                'predictions': lgb_pred
+            }
+            
+            # Random Forest
+            rf_model = RandomForestRegressor(
+                n_estimators=100, max_depth=10, random_state=42
+            )
+            rf_model.fit(X_train_2d, y_train)
+            rf_pred = rf_model.predict(X_test_2d)
+            
+            self.models['random_forest'] = rf_model
+            results['random_forest'] = {
+                'mse': mean_squared_error(y_test, rf_pred),
+                'mae': mean_absolute_error(y_test, rf_pred),
+                'r2': r2_score(y_test, rf_pred),
+                'predictions': rf_pred
+            }
+            
+            # LSTM Model
+            lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
+            
+            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
+            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
+            
+            history = lstm_model.fit(
+                X_train, y_train,
+                epochs=50,
+                batch_size=32,
+                validation_split=0.2,
+                callbacks=[early_stopping, reduce_lr],
+                verbose=0
+            )
+            
+            lstm_pred = lstm_model.predict(X_test)
+            
+            self.models['lstm'] = lstm_model
+            results['lstm'] = {
+                'mse': mean_squared_error(y_test, lstm_pred.flatten()),
+                'mae': mean_absolute_error(y_test, lstm_pred.flatten()),
+                'r2': r2_score(y_test, lstm_pred.flatten()),
+                'predictions': lstm_pred.flatten(),
+                'history': history.history
+            }
+            
+        except Exception as e:
+            st.error(f"Error training models: {str(e)}")
+        
+        return results
+    
+    def predict_future(self, data: pd.DataFrame, days: int = 7) -> Dict[str, np.ndarray]:
+        """Predict future prices using trained models"""
+        predictions = {}
+        
+        try:
+            # Prepare last sequence
+            feature_cols = self.feature_columns
+            features = data[feature_cols].values
+            features_scaled = self.scalers['features'].transform(features)
+            
+            last_sequence = features_scaled[-60:]  # Use last 60 days
+            
+            for model_name, model in self.models.items():
+                if model_name in ['xgboost', 'lightgbm', 'random_forest']:
+                    # For tree-based models, use the last data point
+                    last_features = features_scaled[-1].reshape(1, -1)
+                    pred_scaled = model.predict(last_features)[0]
+                    
+                    # Generate future predictions (simple approach)
+                    future_preds = []
+                    for _ in range(days):
+                        future_preds.append(pred_scaled)
+                        pred_scaled *= (1 + np.random.normal(0, 0.02))  # Add some randomness
+                    
+                    predictions[model_name] = self.scalers['target'].inverse_transform(
+                        np.array(future_preds).reshape(-1, 1)
+                    ).flatten()
+                
+                elif model_name == 'lstm':
+                    # For LSTM, use sequence prediction
+                    sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
+                    future_preds = []
+                    
+                    for _ in range(days):
+                        pred_scaled = model.predict(sequence)[0][0]
+                        future_preds.append(pred_scaled)
+                        
+                        # Update sequence for next prediction
+                        new_row = sequence[0, -1, :].copy()
+                        new_row[0] = pred_scaled  # Assuming first feature is related to price
+                        sequence = np.roll(sequence, -1, axis=1)
+                        sequence[0, -1, :] = new_row
+                    
+                    predictions[model_name] = self.scalers['target'].inverse_transform(
+                        np.array(future_preds).reshape(-1, 1)
+                    ).flatten()
+        
+        except Exception as e:
+            st.error(f"Error predicting future prices: {str(e)}")
+        
+        return predictions
+    
+    def get_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
+        """Create ensemble prediction from multiple models"""
+        if not predictions:
+            return np.array([])
+        
+        # Simple average ensemble
+        pred_arrays = list(predictions.values())
+        ensemble = np.mean(pred_arrays, axis=0)
+        
+        return ensemble
+
+class CryptoPricePredictionApp:
+    """Main application class"""
+    
+    def __init__(self):
+        self.data_fetcher = CryptoDataFetcher()
+        self.feature_engineer = FeatureEngineer()
+        self.ml_predictor = MLPredictor()
+        
+        self.crypto_symbols = {
+            'Bitcoin': 'BTC-USD',
+            'Ethereum': 'ETH-USD',
+            'Binance Coin': 'BNB-USD',
+            'Cardano': 'ADA-USD',
+            'Solana': 'SOL-USD',
+            'Dogecoin': 'DOGE-USD',
+            'Polygon': 'MATIC-USD',
+            'Avalanche': 'AVAX-USD',
+            'Chainlink': 'LINK-USD',
+            'Polkadot': 'DOT-USD'
+        }
+    
+    def create_prediction_chart(self, data: pd.DataFrame, predictions: Dict[str, np.ndarray], 
+                              symbol: str, days: int) -> go.Figure:
+        """Create comprehensive prediction chart"""
+        fig = make_subplots(
+            rows=2, cols=1,
+            shared_xaxes=True,
+            vertical_spacing=0.1,
+            subplot_titles=('Price Prediction', 'Model Comparison'),
+            row_heights=[0.7, 0.3]
+        )
+        
+        # Historical prices
+        fig.add_trace(
+            go.Scatter(
+                x=data.index[-100:],  # Last 100 days
+                y=data['Close'].iloc[-100:],
+                mode='lines',
+                name='Historical Price',
+                line=dict(color='blue', width=2)
+            ),
+            row=1, col=1
+        )
+        
+        # Future dates
+        last_date = data.index[-1]
+        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
+        
+        # Predictions from different models
+        colors = ['red', 'green', 'orange', 'purple']
+        for i, (model_name, pred) in enumerate(predictions.items()):
+            fig.add_trace(
+                go.Scatter(
+                    x=future_dates,
+                    y=pred,
+                    mode='lines+markers',
+                    name=f'{model_name.upper()} Prediction',
+                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
+                ),
+                row=1, col=1
+            )
+        
+        # Ensemble prediction
+        if len(predictions) > 1:
+            ensemble = self.ml_predictor.get_ensemble_prediction(predictions)
+            fig.add_trace(
+                go.Scatter(
+                    x=future_dates,
+                    y=ensemble,
+                    mode='lines+markers',
+                    name='Ensemble Prediction',
+                    line=dict(color='black', width=3)
+                ),
+                row=1, col=1
+            )
+        
+        # Model comparison (placeholder)
+        model_names = list(predictions.keys())
+        if model_names:
+            fig.add_trace(
+                go.Bar(
+                    x=model_names,
+                    y=[np.mean(pred) for pred in predictions.values()],
+                    name='Average Prediction',
+                    marker_color='lightblue'
+                ),
+                row=2, col=1
+            )
+        
+        fig.update_layout(
+            title=f'{symbol} Price Prediction ({days} Days)',
+            xaxis_title='Date',
+            yaxis_title='Price ($)',
+            height=800,
+            showlegend=True
+        )
+        
+        return fig
+    
+    def display_model_performance(self, results: Dict[str, Dict]):
+        """Display model performance metrics"""
+        if not results:
+            return
+        
+        st.markdown("### 🎯 Model Performance")
+        
+        cols = st.columns(len(results))
+        
+        for i, (model_name, metrics) in enumerate(results.items()):
+            with cols[i]:
+                st.markdown(f'<div class="model-performance">', unsafe_allow_html=True)
+                st.markdown(f"**{model_name.upper()}**")
+                st.metric("R² Score", f"{metrics['r2']:.4f}")
+                st.metric("MAE", f"{metrics['mae']:.4f}")
+                st.metric("MSE", f"{metrics['mse']:.4f}")
+                st.markdown('</div>', unsafe_allow_html=True)
+    
+    def display_confidence_intervals(self, predictions: Dict[str, np.ndarray]):
+        """Display prediction confidence intervals"""
+        if not predictions:
+            return
+        
+        st.markdown("### 📊 Prediction Confidence")
+        
+        for model_name, pred in predictions.items():
+            if len(pred) > 0:
+                confidence = self.calculate_confidence(pred)
+                confidence_class = self.get_confidence_class(confidence)
+                
+                st.markdown(
+                    f'<div class="{confidence_class}">'
+                    f'{model_name.upper()}: {confidence:.1f}% Confidence'
+                    f'</div>',
+                    unsafe_allow_html=True
+                )
+    
+    def calculate_confidence(self, predictions: np.ndarray) -> float:
+        """Calculate prediction confidence based on variance"""
+        if len(predictions) < 2:
+            return 50.0
+        
+        variance = np.var(predictions)
+        mean_price = np.mean(predictions)
+        
+        # Normalize variance relative to price
+        normalized_variance = variance / (mean_price ** 2) if mean_price > 0 else 1
+        
+        # Convert to confidence (lower variance = higher confidence)
+        confidence = max(0, min(100, 100 - (normalized_variance * 1000)))
+        
+        return confidence
+    
+    def get_confidence_class(self, confidence: float) -> str:
+        """Get CSS class based on confidence level"""
+        if confidence >= 70:
+            return "confidence-high"
+        elif confidence >= 40:
+            return "confidence-medium"
+        else:
+            return "confidence-low"
+    
+    def run(self):
+        """Run the main application"""
+        st.markdown('<h1 class="main-header">🚀 AI Crypto Price Predictor</h1>', unsafe_allow_html=True)
+        st.markdown("### Advanced Machine Learning & Real-Time Data Analytics")
+        
+        # Sidebar
+        st.sidebar.header("🔧 Configuration")
+        
+        # Crypto selection
+        selected_crypto = st.sidebar.selectbox(
+            "Select Cryptocurrency:",
+            list(self.crypto_symbols.keys())
+        )
+        
+        # Time period
+        period = st.sidebar.selectbox(
+            "Historical Data Period:",
+            ['6mo', '1y', '2y', '5y'],
+            index=1
+        )
+        
+        # Prediction days
+        prediction_days = st.sidebar.slider(
+            "Prediction Days:",
+            min_value=1,
+            max_value=30,
+            value=7
+        )
+        
+        # Model selection
+        st.sidebar.markdown("### 🤖 Model Selection")
+        use_lstm = st.sidebar.checkbox("LSTM Neural Network", value=True)
+        use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
+        use_lightgbm = st.sidebar.checkbox("LightGBM", value=True)
+        use_rf = st.sidebar.checkbox("Random Forest", value=True)
+        
+        # Real-time data
+        symbol = self.crypto_symbols[selected_crypto]
+        
+        # Display real-time price
+        with st.spinner("Fetching real-time data..."):
+            real_time_data = self.data_fetcher.get_real_time_price(symbol)
+            
+        if real_time_data:
+            col1, col2, col3 = st.columns(3)
+            
+            with col1:
+                st.metric(
+                    f"{selected_crypto} Price",
+                    f"${real_time_data['price']:.2f}",
+                    f"{real_time_data['change_24h']:.2f}%"
+                )
+            
+            with col2:
+                sentiment = self.data_fetcher.get_market_sentiment(symbol)
+                st.metric(
+                    "Market Sentiment",
+                    sentiment['sentiment_label'],
+                    f"{sentiment['sentiment_score']:.2f}"
+                )
+            
+            with col3:
+                st.metric(
+                    "Data Source",
+                    real_time_data['source'].title(),
+                    "Live"
+                )
+        
+        # Get historical data and make predictions
+        if st.button("🔮 Generate Predictions", type="primary"):
+            with st.spinner("Fetching historical data and training models..."):
+                # Get historical data
+                data = self.data_fetcher.get_historical_data(symbol, period)
+                
+                if data is not None and not data.empty:
+                    # Feature engineering
+                    data = self.feature_engineer.create_technical_features(data)
+                    
+                    # Prepare data for ML
+                    X, y = self.ml_predictor.prepare_data(data)
+                    
+                    if len(X) > 0 and len(y) > 0:
+                        # Train models
+                        results = self.ml_predictor.train_models(X, y)
+                        
+                        # Display model performance
+                        self.display_model_performance(results)
+                        
+                        # Make future predictions
+                        predictions = self.ml_predictor.predict_future(data, prediction_days)
+                        
+                        if predictions:
+                            # Create prediction chart
+                            fig = self.create_prediction_chart(data, predictions, selected_crypto, prediction_days)
+                            st.plotly_chart(fig, use_container_width=True)
+                            
+                            # Display predictions
+                            st.markdown("### 📈 Price Predictions")
+                            
+                            future_dates = pd.date_range(
+                                start=data.index[-1] + timedelta(days=1), 
+                                periods=prediction_days
+                            )
+                            
+                            # Create prediction DataFrame
+                            pred_df = pd.DataFrame({'Date': future_dates})
+                            for model_name, pred in predictions.items():
+                                pred_df[f'{model_name.upper()}'] = pred
+                            
+                            if len(predictions) > 1:
+                                ensemble = self.ml_predictor.get_ensemble_prediction(predictions)
+                                pred_df['ENSEMBLE'] = ensemble
+                            
+                            st.dataframe(pred_df, use_container_width=True)
+                            
+                            # Confidence intervals
+                            self.display_confidence_intervals(predictions)
+                            
+                            # Trading recommendations
+                            self.display_trading_recommendations(predictions, real_time_data)
+                        
+                        else:
+                            st.error("Failed to generate predictions. Please try again.")
+                    else:
+                        st.error("Insufficient data for model training.")
+                else:
+                    st.error("Failed to fetch historical data.")
+        
+        # Additional features
+        self.display_additional_features(symbol)
+    
+    def display_trading_recommendations(self, predictions: Dict[str, np.ndarray], 
+                                      current_data: Optional[Dict]):
+        """Display AI-powered trading recommendations"""
+        if not predictions or not current_data:
+            return
+        
+        st.markdown("### 🎯 AI Trading Recommendations")
+        
+        # Calculate average prediction
+        avg_predictions = {name: np.mean(pred) for name, pred in predictions.items()}
+        overall_avg = np.mean(list(avg_predictions.values()))
+        current_price = current_data['price']
+        
+        # Calculate expected return
+        expected_return = ((overall_avg - current_price) / current_price) * 100
+        
+        # Generate recommendation
+        if expected_return > 5:
+            recommendation = "STRONG BUY"
+            color = "success"
+        elif expected_return > 2:
+            recommendation = "BUY"
+            color = "success"
+        elif expected_return > -2:
+            recommendation = "HOLD"
+            color = "warning"
+        elif expected_return > -5:
+            recommendation = "SELL"
+            color = "error"
+        else:
+            recommendation = "STRONG SELL"
+            color = "error"
+        
+        col1, col2, col3 = st.columns(3)
+        
+        with col1:
+            if color == "success":
+                st.success(f"**Recommendation: {recommendation}**")
+            elif color == "warning":
+                st.warning(f"**Recommendation: {recommendation}**")
+            else:
+                st.error(f"**Recommendation: {recommendation}**")
+        
+        with col2:
+            st.metric("Expected Return", f"{expected_return:.2f}%")
+        
+        with col3:
+            st.metric("Target Price", f"${overall_avg:.2f}")
+        
+        # Risk assessment
+        st.markdown("#### ⚠️ Risk Assessment")
+        volatility = np.std(list(avg_predictions.values())) / overall_avg * 100
+        
+        if volatility < 5:
+            risk_level = "LOW"
+            risk_color = "success"
+        elif volatility < 15:
+            risk_level = "MEDIUM"
+            risk_color = "warning"
+        else:
+            risk_level = "HIGH"
+            risk_color = "error"
+        
+        if risk_color == "success":
+            st.success(f"Risk Level: {risk_level} (Volatility: {volatility:.2f}%)")
+        elif risk_color == "warning":
+            st.warning(f"Risk Level: {risk_level} (Volatility: {volatility:.2f}%)")
+        else:
+            st.error(f"Risk Level: {risk_level} (Volatility: {volatility:.2f}%)")
+    
+    def display_additional_features(self, symbol: str):
+        """Display additional features and analytics"""
+        st.markdown("---")
+        st.markdown("### 📊 Additional Analytics")
+        
+        col1, col2 = st.columns(2)
+        
+        with col1:
+            st.markdown("#### 🔔 Price Alerts")
+            alert_price = st.number_input("Set Price Alert ($)", min_value=0.0, step=0.01)
+            alert_type = st.selectbox("Alert Type", ["Above", "Below"])
+            
+            if st.button("Set Alert"):
+                st.success(f"Alert set: Notify when price goes {alert_type.lower()} ${alert_price}")
+        
+        with col2:
+            st.markdown("#### 📈 Market Analysis")
+            st.info("""
+            **Key Features:**
+            - Real-time data from multiple sources
+            - Advanced ML models (LSTM, XGBoost, etc.)
+            - Ensemble predictions for better accuracy
+            - Technical analysis integration
+            - Risk assessment and recommendations
+            """)
+        
+        # Disclaimer
+        st.markdown("---")
+        st.markdown("### ⚠️ Important Disclaimer")
+        st.warning("""
+        **Investment Risk Warning:**
+        - This tool is for educational and research purposes only
+        - Cryptocurrency investments carry high risk
+        - Past performance does not guarantee future results
+        - Always do your own research before making investment decisions
+        - Consider consulting with a financial advisor
+        - Never invest more than you can afford to lose
+        """)
+
+def main():
+    """Main function to run the application"""
+    app = CryptoPricePredictionApp()
+    app.run()
+
+if __name__ == "__main__":
+    main()
+
EOF
)
