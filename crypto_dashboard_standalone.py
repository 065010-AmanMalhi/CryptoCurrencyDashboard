# crypto_dashboard.py - Fixed Version

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .crypto-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process cryptocurrency data with robust error handling"""
    
    possible_paths = [
        "top_100_cryptos_with_correct_network.csv",
        "data/top_100_cryptos_with_correct_network.csv", 
        "../data/top_100_cryptos_with_correct_network.csv",
        "./top_100_cryptos_with_correct_network.csv"
    ]
    
    df = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        except Exception as e:
            continue
    
    if df is None:
        st.error("Data file not found. Please check file location.")
        st.stop()
    
    # Debug: Show actual columns
    st.sidebar.write("Available columns:", list(df.columns))
    
    # Convert date column (try different possible column names)
    date_columns = ['date', 'Date', 'timestamp', 'time']
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df['date'] = pd.to_datetime(df[date_col])
    else:
        st.error("No date column found!")
        st.stop()
    
    # Calculate features only if base columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", list(df.columns))
        st.stop()
    
    # Safe feature calculation
    try:
        # Daily Return Percentage
        df["Daily_Return_%"] = ((df["close"] - df["open"]) / df["open"]) * 100
        df["Daily_Return_%"] = df["Daily_Return_%"].fillna(0)
        
        # Price Change
        df["Price_Change"] = df["close"] - df["open"]
        
        # High-Low Range
        df["High_Low_Range"] = df["high"] - df["low"]
        
        # Body Size
        df["Body_Size"] = abs(df["close"] - df["open"])
        
        # Candle Type
        df["Candle_Type"] = df.apply(lambda row: 
            "Bullish" if row["close"] > row["open"]
            else ("Bearish" if row["close"] < row["open"] else "Neutral"), 
            axis=1
        )
        
        # Close Position (handle division by zero)
        high_low_diff = df["high"] - df["low"]
        df["Close_Position"] = np.where(
            high_low_diff == 0,
            0.5,
            (df["close"] - df["low"]) / high_low_diff
        )
        
        # Volatility
        df["Volatility_%"] = ((df["high"] - df["low"]) / df["open"]) * 100
        df["Volatility_%"] = df["Volatility_%"].fillna(0)
        
        # Sort by symbol and date for cumulative calculations
        df = df.sort_values(['symbol', 'date'])
        
        # Cumulative Return (by symbol)
        df['Cumulative_Return'] = df.groupby('symbol')['Daily_Return_%'].transform(
            lambda x: (1 + x/100).cumprod()
        )
        
    except Exception as e:
        st.error(f"Error calculating features: {str(e)}")
        st.write("Please check your data format")
        st.stop()
    
    return df

# Load Data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Header
st.markdown("""
<div class="crypto-header">
    <h1>📈 Cryptocurrency Dashboard</h1>
    <p>Advanced analytics and visualization for cryptocurrency market data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Data Filters")

# Check if required columns exist
if 'symbol' not in df.columns:
    st.error("No 'symbol' column found in data")
    st.stop()

# Dataset overview
with st.sidebar.expander("📊 Dataset Overview", expanded=True):
    st.write(f"**Total Records:** {len(df):,}")
    if 'symbol' in df.columns:
        st.write(f"**Unique Cryptocurrencies:** {df['symbol'].nunique()}")
        # Show all available cryptocurrencies
        st.write("**Available Coins:**")
        st.write(", ".join(sorted(df['symbol'].unique())[:20]) + "..." if len(df['symbol'].unique()) > 20 else ", ".join(sorted(df['symbol'].unique())))
    if 'network' in df.columns:
        st.write(f"**Blockchain Networks:** {df['network'].nunique()}")
    st.write(f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Cryptocurrency selection
available_symbols = sorted(df['symbol'].unique())

# Show popular coins first if they exist
popular_coins = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 'LTCUSDT']
default_coins = [coin for coin in popular_coins if coin in available_symbols]
if not default_coins:
    default_coins = available_symbols[:5]  # Fallback to first 5

selected_symbols = st.sidebar.multiselect(
    "🪙 Select Cryptocurrencies:",
    options=available_symbols,
    default=default_coins,
    help="Choose which cryptocurrencies to analyze"
)

# Add quick selection buttons
st.sidebar.markdown("**Quick Select:**")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("All Coins"):
        selected_symbols = available_symbols
with col2:
    if st.button("Top 10"):
        selected_symbols = available_symbols[:10]

# Network filter (only if network column exists)
if 'network' in df.columns:
    available_networks = sorted(df['network'].unique())
    selected_networks = st.sidebar.multiselect(
        "🌐 Select Networks:",
        options=available_networks,
        default=available_networks[:5],  # Limit default selection
        help="Filter by blockchain network"
    )
else:
    selected_networks = None

# Date range
min_date = df['date'].min().date()
max_date = df['date'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Apply Filters
filtered_df = df[df['symbol'].isin(selected_symbols)]

if selected_networks and 'network' in df.columns:
    filtered_df = filtered_df[filtered_df['network'].isin(selected_networks)]

filtered_df = filtered_df[
    (filtered_df['date'] >= pd.to_datetime(start_date)) &
    (filtered_df['date'] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    st.warning("No data matches your current filter selection. Please adjust the filters.")
    st.stop()

# Key Metrics
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if 'Daily_Return_%' in filtered_df.columns:
        avg_daily_return = filtered_df['Daily_Return_%'].mean()
        st.metric("Average Daily Return", f"{avg_daily_return:.2f}%")
    else:
        st.metric("Average Daily Return", "N/A")

with col2:
    if 'Volatility_%' in filtered_df.columns:
        avg_volatility = filtered_df['Volatility_%'].mean()
        st.metric("Average Volatility", f"{avg_volatility:.2f}%")
    else:
        st.metric("Average Volatility", "N/A")

with col3:
    st.metric("Trading Days", f"{len(filtered_df):,}")

with col4:
    if 'Candle_Type' in filtered_df.columns:
        bullish_percentage = (filtered_df['Candle_Type'] == 'Bullish').mean() * 100
        st.metric("Bullish Days", f"{bullish_percentage:.1f}%")
    else:
        st.metric("Bullish Days", "N/A")

with col5:
    if 'Daily_Return_%' in filtered_df.columns:
        max_daily_gain = filtered_df['Daily_Return_%'].max()
        st.metric("Max Daily Gain", f"{max_daily_gain:.2f}%")
    else:
        st.metric("Max Daily Gain", "N/A")

# Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🕯️ Candlestick Charts", "📊 Returns Analysis", "📋 Data Explorer"])

# Tab 1: Price Analysis
with tab1:
    st.subheader("Price Movement Over Time")
    
    # Basic price chart
    fig_price = px.line(
        filtered_df,
        x='date',
        y='close', 
        color='symbol',
        title='Closing Prices Over Time',
        labels={'close': 'Price (USD)', 'date': 'Date'}
    )
    fig_price.update_layout(height=500)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volume analysis if we have the data
    if 'High_Low_Range' in filtered_df.columns:
        fig_range = px.line(
            filtered_df,
            x='date',
            y='High_Low_Range',
            color='symbol', 
            title='Daily Price Range (High - Low)',
            labels={'High_Low_Range': 'Price Range (USD)'}
        )
        fig_range.update_layout(height=400)
        st.plotly_chart(fig_range, use_container_width=True)

# Tab 2: Candlestick Charts
with tab2:
    st.subheader("🕯️ Candlestick Analysis")
    
    # Show first 2 symbols to avoid performance issues
    symbols_to_show = selected_symbols[:2]
    
    for symbol in symbols_to_show:
        symbol_data = filtered_df[filtered_df['symbol'] == symbol].sort_values('date')
        
        if not symbol_data.empty and len(symbol_data) > 0:
            try:
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=symbol_data['date'],
                    open=symbol_data['open'],
                    high=symbol_data['high'], 
                    low=symbol_data['low'],
                    close=symbol_data['close'],
                    name=symbol
                )])
                
                fig_candlestick.update_layout(
                    title=f'Candlestick Chart - {symbol}',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig_candlestick, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating candlestick chart for {symbol}: {str(e)}")

# Tab 3: Returns Analysis
with tab3:
    st.subheader("📊 Returns and Performance Analysis")
    
    if 'Daily_Return_%' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns distribution
            fig_returns = px.histogram(
                filtered_df,
                x='Daily_Return_%',
                color='symbol',
                title='Daily Returns Distribution',
                labels={'Daily_Return_%': 'Daily Return (%)'}
            )
            fig_returns.update_layout(height=400)
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Cumulative returns if available
            if 'Cumulative_Return' in filtered_df.columns:
                fig_cumulative = px.line(
                    filtered_df,
                    x='date',
                    y='Cumulative_Return',
                    color='symbol',
                    title='Cumulative Returns Over Time',
                    labels={'Cumulative_Return': 'Cumulative Return'}
                )
                fig_cumulative.update_layout(height=400)
                st.plotly_chart(fig_cumulative, use_container_width=True)
    else:
        st.info("Daily return calculations not available")

# Tab 4: Data Explorer
with tab4:
    st.subheader("📋 Data Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col2: 
        st.metric("Date Range", f"{(filtered_df['date'].max() - filtered_df['date'].min()).days} days")
    with col3:
        unique_dates = filtered_df['date'].nunique()
        avg_volume = len(filtered_df) // unique_dates if unique_dates > 0 else 0
        st.metric("Avg Records/Day", f"{avg_volume:,}")
    
    # Display data
    st.subheader("Raw Data Sample")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Download option
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv_data,
        file_name=f"crypto_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666;">
        <p><strong>Crypto Dashboard</strong> • Analyzing {len(filtered_df):,} records</p>
        <p>Data from {filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}</p>
    </div>
    """,
    unsafe_allow_html=True
)
