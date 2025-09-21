# crypto_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        border-radius: 0.375rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .crypto-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.375rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Data Loading Function
# =========================
@st.cache_data
def load_data():
    """Load and process cryptocurrency data with error handling"""
    
    # Try different possible file locations
    possible_paths = [
        "top_100_cryptos_with_correct_network.csv",
        "data/top_100_cryptos_with_correct_network.csv", 
        "../data/top_100_cryptos_with_correct_network.csv",
        "./top_100_cryptos_with_correct_network.csv",
        "/content/drive/MyDrive/Colab Notebooks/top_100_cryptos_with_correct_network.csv"
    ]
    
    df = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"Data loaded from: {path}")
                break
        except Exception as e:
            continue
    
    if df is None:
        st.error("""
        **Data file not found!** 
        
        Please ensure 'top_100_cryptos_with_correct_network.csv' is in one of these locations:
        - Same folder as this script
        - In a 'data/' subfolder
        - Update the file path in the code
        """)
        st.stop()
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate additional features
    df = calculate_features(df)
    
    return df

def calculate_features(df):
    """Calculate technical indicators and features"""
    
    # Daily Return Percentage
    df["Daily_Return_%"] = ((df["close"] - df["open"]) / df["open"]) * 100
    
    # Price Change (absolute)
    df["Price_Change"] = df["close"] - df["open"]
    
    # High-Low Range (daily volatility measure)
    df["High_Low_Range"] = df["high"] - df["low"]
    
    # Body Size (candlestick body)
    df["Body_Size"] = abs(df["close"] - df["open"])
    
    # Candle Type classification
    df["Candle_Type"] = df.apply(lambda row: 
        "Bullish" if row["close"] > row["open"]
        else ("Bearish" if row["close"] < row["open"] else "Neutral"), 
        axis=1
    )
    
    # Close position within daily range
    df["Close_Position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["Close_Position"] = df["Close_Position"].fillna(0.5)  # Handle division by zero
    
    # Daily Volatility Percentage
    df["Volatility_%"] = ((df["high"] - df["low"]) / df["open"]) * 100
    
    # Sort by symbol and date for cumulative calculations
    df = df.sort_values(['symbol', 'date'])
    
    # Cumulative Return (properly calculated by symbol)
    df['Cumulative_Return'] = df.groupby('symbol')['Daily_Return_%'].transform(
        lambda x: (1 + x/100).cumprod()
    )
    
    # Moving averages (if enough data points)
    df['MA_7'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['MA_30'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    
    return df

# =========================
# Load Data
# =========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# =========================
# Header Section
# =========================
st.markdown("""
<div class="crypto-header">
    <h1>📈 Cryptocurrency Dashboard</h1>
    <p>Advanced analytics and visualization for cryptocurrency market data</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🔍 Data Filters")

# Dataset overview
with st.sidebar.expander("📊 Dataset Overview", expanded=False):
    st.write(f"**Total Records:** {len(df):,}")
    st.write(f"**Unique Cryptocurrencies:** {df['symbol'].nunique()}")
    st.write(f"**Blockchain Networks:** {df['network'].nunique()}")
    st.write(f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Cryptocurrency selection
available_symbols = sorted(df['symbol'].unique())
selected_symbols = st.sidebar.multiselect(
    "🪙 Select Cryptocurrencies:",
    options=available_symbols,
    default=available_symbols[:5],  # Default to first 5
    help="Choose which cryptocurrencies to analyze"
)

# Network filter
available_networks = sorted(df['network'].unique())
selected_networks = st.sidebar.multiselect(
    "🌐 Select Networks:",
    options=available_networks,
    default=available_networks,
    help="Filter by blockchain network"
)

# Date range selection
min_date = df['date'].min().date()
max_date = df['date'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# Price range filter
price_min = float(df['close'].min())
price_max = float(df['close'].max())
price_range = st.sidebar.slider(
    "💰 Price Range (USD)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    format="$%.2f"
)

# Advanced filters
with st.sidebar.expander("⚙️ Advanced Filters", expanded=False):
    volatility_threshold = st.slider(
        "Max Daily Volatility (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Filter out extremely volatile days"
    )
    
    candle_type_filter = st.multiselect(
        "Candle Types",
        options=['Bullish', 'Bearish', 'Neutral'],
        default=['Bullish', 'Bearish', 'Neutral']
    )

# =========================
# Apply Filters
# =========================
filtered_df = df[
    (df['symbol'].isin(selected_symbols)) &
    (df['network'].isin(selected_networks)) &
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date)) &
    (df['close'] >= price_range[0]) &
    (df['close'] <= price_range[1]) &
    (df['Volatility_%'] <= volatility_threshold) &
    (df['Candle_Type'].isin(candle_type_filter))
]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("⚠️ No data matches your current filter selection. Please adjust the filters.")
    st.stop()

# =========================
# Key Performance Metrics
# =========================
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_daily_return = filtered_df['Daily_Return_%'].mean()
    st.metric(
        label="Average Daily Return",
        value=f"{avg_daily_return:.2f}%",
        delta=f"{avg_daily_return - df['Daily_Return_%'].mean():.2f}%"
    )

with col2:
    avg_volatility = filtered_df['Volatility_%'].mean()
    st.metric(
        label="Average Volatility", 
        value=f"{avg_volatility:.2f}%"
    )

with col3:
    total_trading_days = len(filtered_df)
    st.metric(
        label="Trading Days",
        value=f"{total_trading_days:,}"
    )

with col4:
    bullish_percentage = (filtered_df['Candle_Type'] == 'Bullish').mean() * 100
    st.metric(
        label="Bullish Days",
        value=f"{bullish_percentage:.1f}%"
    )

with col5:
    max_daily_gain = filtered_df['Daily_Return_%'].max()
    st.metric(
        label="Max Daily Gain",
        value=f"{max_daily_gain:.2f}%"
    )

# =========================
# Main Dashboard Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price Analysis", 
    "🕯️ Candlestick Charts", 
    "📊 Returns & Volatility", 
    "🔥 Correlation Analysis",
    "🎯 Network Analysis",
    "📋 Data Explorer"
])

# =========================
# Tab 1: Price Analysis
# =========================
with tab1:
    st.subheader("Price Movement Over Time")
    
    # Main price chart
    fig_price = px.line(
        filtered_df,
        x='date',
        y='close', 
        color='symbol',
        title='Closing Prices Over Time',
        labels={'close': 'Price (USD)', 'date': 'Date'},
        hover_data=['open', 'high', 'low', 'Daily_Return_%']
    )
    fig_price.update_layout(
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Price with moving averages (if single symbol selected)
    if len(selected_symbols) == 1:
        symbol_data = filtered_df[filtered_df['symbol'] == selected_symbols[0]]
        
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=symbol_data['date'], 
            y=symbol_data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=symbol_data['date'],
            y=symbol_data['MA_7'], 
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=1)
        ))
        fig_ma.add_trace(go.Scatter(
            x=symbol_data['date'],
            y=symbol_data['MA_30'],
            mode='lines', 
            name='30-Day MA',
            line=dict(color='red', width=1)
        ))
        
        fig_ma.update_layout(
            title=f'{selected_symbols[0]} - Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_ma, use_container_width=True)

# =========================
# Tab 2: Candlestick Charts  
# =========================
with tab2:
    st.subheader("🕯️ Candlestick Analysis")
    
    # Limit to first 3 symbols for performance
    symbols_to_show = selected_symbols[:3]
    
    for symbol in symbols_to_show:
        symbol_data = filtered_df[filtered_df['symbol'] == symbol].sort_values('date')
        
        if not symbol_data.empty:
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
                xaxis_rangeslider_visible=False,
                showlegend=False
            )
            
            st.plotly_chart(fig_candlestick, use_container_width=True)
    
    if len(selected_symbols) > 3:
        st.info(f"Showing candlestick charts for first 3 symbols. {len(selected_symbols) - 3} more symbols available in your selection.")

# =========================
# Tab 3: Returns & Volatility
# =========================
with tab3:
    st.subheader("📊 Returns and Volatility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily returns distribution
        fig_returns_dist = px.histogram(
            filtered_df,
            x='Daily_Return_%',
            color='symbol',
            title='Daily Returns Distribution',
            labels={'Daily_Return_%': 'Daily Return (%)'},
            marginal='box',
            nbins=50
        )
        fig_returns_dist.update_layout(height=400)
        st.plotly_chart(fig_returns_dist, use_container_width=True)
        
        # Volatility over time
        fig_volatility = px.line(
            filtered_df,
            x='date',
            y='Volatility_%',
            color='symbol',
            title='Daily Volatility Over Time',
            labels={'Volatility_%': 'Volatility (%)'}
        )
        fig_volatility.update_layout(height=400)
        st.plotly_chart(fig_volatility, use_container_width=True)
    
    with col2:
        # Risk vs Return scatter
        risk_return = filtered_df.groupby('symbol').agg({
            'Daily_Return_%': 'mean',
            'Volatility_%': 'mean',
            'close': 'count'
        }).reset_index()
        risk_return.columns = ['symbol', 'avg_return', 'avg_volatility', 'data_points']
        
        fig_risk_return = px.scatter(
            risk_return,
            x='avg_return',
            y='avg_volatility',
            size='data_points',
            hover_name='symbol',
            title='Risk vs Return Profile',
            labels={
                'avg_return': 'Average Daily Return (%)',
                'avg_volatility': 'Average Volatility (%)'
            }
        )
        fig_risk_return.update_layout(height=400)
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        # Cumulative returns
        if 'Cumulative_Return' in filtered_df.columns:
            fig_cumulative = px.line(
                filtered_df,
                x='date',
                y='Cumulative_Return',
                color='symbol',
                title='Cumulative Returns Over Time',
                labels={'Cumulative_Return': 'Cumulative Return (Multiple)'},
                log_y=True
            )
            fig_cumulative.update_layout(height=400)
            st.plotly_chart(fig_cumulative, use_container_width=True)

# =========================
# Tab 4: Correlation Analysis
# =========================
with tab4:
    st.subheader("🔥 Correlation and Relationship Analysis")
    
    # Feature correlation heatmap
    numeric_features = [
        'open', 'high', 'low', 'close', 
        'Daily_Return_%', 'Volatility_%', 'Body_Size', 'Close_Position'
    ]
    available_features = [col for col in numeric_features if col in filtered_df.columns]
    
    if len(available_features) > 1:
        correlation_matrix = filtered_df[available_features].corr()
        
        fig_correlation = px.imshow(
            correlation_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto=True
        )
        fig_correlation.update_layout(height=500)
        st.plotly_chart(fig_correlation, use_container_width=True)
    
    # Price correlation between symbols (if multiple selected)
    if len(selected_symbols) > 1:
        price_pivot = filtered_df.pivot_table(
            values='close', 
            index='date', 
            columns='symbol', 
            fill_value=np.nan
        )
        
        if not price_pivot.empty:
            price_corr = price_pivot.corr()
            
            fig_price_corr = px.imshow(
                price_corr,
                title='Price Correlation Between Selected Cryptocurrencies',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig_price_corr.update_layout(height=400)
            st.plotly_chart(fig_price_corr, use_container_width=True)

# =========================
# Tab 5: Network Analysis
# =========================
with tab5:
    st.subheader("🎯 Blockchain Network Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Network performance summary
        network_stats = filtered_df.groupby('network').agg({
            'Daily_Return_%': ['mean', 'std'],
            'Volatility_%': 'mean',
            'symbol': 'nunique',
            'close': 'count'
        }).round(2)
        
        network_stats.columns = ['Avg_Return', 'Return_Std', 'Avg_Volatility', 'Unique_Coins', 'Total_Records']
        network_stats = network_stats.reset_index()
        
        fig_network_performance = px.scatter(
            network_stats,
            x='Avg_Return',
            y='Avg_Volatility', 
            size='Total_Records',
            hover_name='network',
            title='Network Performance: Return vs Volatility',
            labels={
                'Avg_Return': 'Average Daily Return (%)',
                'Avg_Volatility': 'Average Volatility (%)'
            }
        )
        st.plotly_chart(fig_network_performance, use_container_width=True)
    
    with col2:
        # Candle type distribution by network
        candle_network = filtered_df.groupby(['network', 'Candle_Type']).size().reset_index(name='count')
        candle_network['percentage'] = candle_network.groupby('network')['count'].transform(lambda x: x / x.sum() * 100)
        
        fig_candle_network = px.bar(
            candle_network,
            x='network',
            y='percentage',
            color='Candle_Type',
            title='Candle Type Distribution by Network',
            labels={'percentage': 'Percentage (%)'}
        )
        fig_candle_network.update_xaxis(tickangle=45)
        st.plotly_chart(fig_candle_network, use_container_width=True)
    
    # Network statistics table
    st.subheader("Network Statistics Summary")
    st.dataframe(network_stats, use_container_width=True)

# =========================
# Tab 6: Data Explorer
# =========================
with tab6:
    st.subheader("📋 Data Explorer")
    
    # Data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col2: 
        st.metric("Date Range", f"{(filtered_df['date'].max() - filtered_df['date'].min()).days} days")
    with col3:
        st.metric("Avg Daily Volume", f"{len(filtered_df) // filtered_df['date'].nunique():,}")
    
    # Interactive data table
    st.subheader("Filtered Dataset")
    
    # Column selection for display
    all_columns = filtered_df.columns.tolist()
    display_columns = st.multiselect(
        "Select columns to display:",
        options=all_columns,
        default=['symbol', 'date', 'open', 'high', 'low', 'close', 'Daily_Return_%', 'Volatility_%', 'network']
    )
    
    if display_columns:
        # Sort options
        sort_column = st.selectbox("Sort by:", options=display_columns, index=1 if 'date' in display_columns else 0)
        sort_ascending = st.checkbox("Sort ascending", value=False)
        
        # Display data
        display_data = filtered_df[display_columns].sort_values(sort_column, ascending=sort_ascending)
        st.dataframe(display_data, use_container_width=True, height=400)
        
        # Download option
        csv_data = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"crypto_data_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Crypto Dashboard</strong> • Built with Streamlit & Plotly</p>
        <p>Displaying {len(filtered_df):,} records from {filtered_df['date'].min().strftime('%Y-%m-%d')} 
        to {filtered_df['date'].max().strftime('%Y-%m-%d')}</p>
        <p>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """,
    unsafe_allow_html=True
)