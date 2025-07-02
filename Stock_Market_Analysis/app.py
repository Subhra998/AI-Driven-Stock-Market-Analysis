import os
import warnings
import asyncio
import sys
import requests

# Set the Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Suppress TensorFlow and PyTorch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import streamlit as st
from utils.data_fetcher import get_stock_data, get_news_data
from utils.technical_analysis import calculate_technical_indicators
from utils.sentiment_analysis import analyze_sentiment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone
import yfinance as yf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import plotly.express as px

# App configuration
st.set_page_config(
    page_title="Indian Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css = """
    body {
        background-color: black;
        color: white;
    }
    .stMetric {
        background-color: #333333;
        border-radius: 10px;
        padding: 10px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# Sidebar
def sidebar():
    st.sidebar.header("Stock Selection")
    
    # List of Nifty 50 tickers and their sectors
    nifty_50_tickers = {
        "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "INFY.NS": "IT", "ICICIBANK.NS": "Banking",
        "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "KOTAKBANK.NS": "Banking", "SBIN.NS": "Banking", "BHARTIARTL.NS": "Telecom",
        "ADANIENT.NS": "Infrastructure", "LT.NS": "Infrastructure", "AXISBANK.NS": "Banking", "ASIANPAINT.NS": "Paints", "BAJFINANCE.NS": "Finance",
        "HCLTECH.NS": "IT", "MARUTI.NS": "Automobile", "SUNPHARMA.NS": "Pharma", "TITAN.NS": "Jewelry", "ULTRACEMCO.NS": "Cement",
        "WIPRO.NS": "IT", "ONGC.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy", "COALINDIA.NS": "Energy",
        "JSWSTEEL.NS": "Steel", "TATAMOTORS.NS": "Automobile", "ADANIPORTS.NS": "Infrastructure", "GRASIM.NS": "Cement", "BPCL.NS": "Energy",
        "INDUSINDBK.NS": "Banking", "DIVISLAB.NS": "Pharma", "EICHERMOT.NS": "Automobile", "HEROMOTOCO.NS": "Automobile", "DRREDDY.NS": "Pharma",
        "BRITANNIA.NS": "FMCG", "APOLLOHOSP.NS": "Healthcare", "CIPLA.NS": "Pharma", "BAJAJFINSV.NS": "Finance", "HDFCLIFE.NS": "Insurance",
        "SBILIFE.NS": "Insurance", "TECHM.NS": "IT", "TATACONSUM.NS": "FMCG", "M&M.NS": "Automobile", "SHREECEM.NS": "Cement",
        "UPL.NS": "Agrochemicals", "ICICIPRULI.NS": "Insurance", "PIDILITIND.NS": "Chemicals", "DLF.NS": "Real Estate", "BAJAJ-AUTO.NS": "Automobile"
    }
    
    sectors = list(set(nifty_50_tickers.values()))
    sectors.sort()
    
    # Sector selection
    sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)
    
    # Filter stocks based on the selected sector
    if sector == "All":
        filtered_tickers = list(nifty_50_tickers.keys())
    else:
        filtered_tickers = [ticker for ticker, sec in nifty_50_tickers.items() if sec == sector]
    
    # Stock selection
    ticker = st.sidebar.selectbox(
        "Select Stock",
        ["All"] + filtered_tickers
    )
    
    # Analysis period selection
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1M", "3M", "6M", "1Y", "YTD"],
        index=2
    )
    
    st.sidebar.header("Simple Moving Averages")
    show_sma_9 = st.sidebar.checkbox("9-day SMA", False)
    show_sma_20 = st.sidebar.checkbox("20-day SMA", True)
    show_sma_50 = st.sidebar.checkbox("50-day SMA", False)
    show_sma_100 = st.sidebar.checkbox("100-day SMA", False)
    show_sma_200 = st.sidebar.checkbox("200-day SMA", False)
    
    st.sidebar.header("Exponential Moving Averages")
    show_ema = st.sidebar.checkbox("20-day EMA", False)

    st.sidebar.header("Bollinger Bands")
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", False)

    st.sidebar.header("Relative Strength Index (RSI)")
    show_rsi = st.sidebar.checkbox("Show RSI (14-day)", False)

    st.sidebar.header("MACD (Moving Average Convergence Divergence)")
    show_macd = st.sidebar.checkbox("Show MACD", False)

    st.sidebar.header("Fundamental Analysis")
    show_fundamental = st.sidebar.checkbox("Show Fundamental Analysis", False)
    
    return {
        "sector": sector,
        "ticker": ticker,
        "analysis_period": analysis_period,
        "show_sma_9": show_sma_9,
        "show_sma_20": show_sma_20,
        "show_sma_50": show_sma_50,
        "show_sma_100": show_sma_100,
        "show_sma_200": show_sma_200,
        "show_ema": show_ema,
        "show_bollinger": show_bollinger,
        "show_rsi": show_rsi,
        "show_macd": show_macd,
        "show_fundamental": show_fundamental
    }

# Updated Fundamental Analysis Section
def fetch_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract relevant fundamental data
        last_update = info.get('mostRecentQuarter', None)  # Most recent quarter
        if last_update and isinstance(last_update, (int, float)):
            try:
                # Convert timestamp to a readable date
                last_update = pd.to_datetime(last_update, unit='s').strftime('%Y-%m-%d')
            except Exception:
                last_update = "Invalid date"
        else:
            last_update = "N/A"  # Handle missing or invalid data

        # Define fundamental parameters
        data = {
            "EPS": info.get('trailingEps', 0),
            "P/E": info.get('trailingPE', 0),
            "P/B": info.get('priceToBook', 0),
            "D/E": info.get('debtToEquity', 0),
            "PEG": info.get('pegRatio', 0),
            "EV/EBITDA": info.get('enterpriseToEbitda', 0),
            "Revenue Growth (%)": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            "Net Profit Margin (%)": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            "ROE (%)": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            "ROCE (%)": info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,  # Approximation
            "FCF Trend": "Positive & Growing" if info.get('freeCashflow', 0) > 0 else "Negative",
            "OCF Trend": "Positive & Growing" if info.get('operatingCashflow', 0) > 0 else "Negative",
        }

        # Decision criteria
        decisions = {
            "EPS": "Best" if data["EPS"] > 20 else "Moderate" if 10 <= data["EPS"] <= 20 else "Bad",
            "P/E": "Best" if data["P/E"] < 15 else "Moderate" if 15 <= data["P/E"] <= 25 else "Bad",
            "P/B": "Best" if data["P/B"] < 3 else "Moderate" if 3 <= data["P/B"] <= 5 else "Bad",
            "D/E": "Best" if data["D/E"] < 0.5 else "Moderate" if 0.5 <= data["D/E"] <= 1.5 else "Bad",
            "PEG": "Best" if data["PEG"] < 1.5 else "Moderate" if 1.5 <= data["PEG"] <= 2 else "Bad",
            "EV/EBITDA": "Best" if data["EV/EBITDA"] < 8 else "Moderate" if 8 <= data["EV/EBITDA"] <= 10 else "Bad",
            "Revenue Growth (%)": "Best" if data["Revenue Growth (%)"] > 15 else "Moderate" if 10 <= data["Revenue Growth (%)"] <= 15 else "Bad",
            "Net Profit Margin (%)": "Best" if data["Net Profit Margin (%)"] > 20 else "Moderate" if 10 <= data["Net Profit Margin (%)"] <= 20 else "Bad",
            "ROE (%)": "Best" if data["ROE (%)"] > 15 else "Moderate" if 10 <= data["ROE (%)"] <= 15 else "Bad",
            "ROCE (%)": "Best" if data["ROCE (%)"] > 15 else "Moderate" if 10 <= data["ROCE (%)"] <= 15 else "Bad",
            "FCF Trend": "Best" if data["FCF Trend"] == "Positive & Growing" else "Bad",
            "OCF Trend": "Best" if data["OCF Trend"] == "Positive & Growing" else "Bad",
        }

        # Count decision categories
        decision_counts = {"Best": 0, "Moderate": 0, "Bad": 0}
        for decision in decisions.values():
            decision_counts[decision] += 1

        # Final classification
        if decision_counts["Best"] > decision_counts["Moderate"] and decision_counts["Best"] > decision_counts["Bad"]:
            classification = "Best (Strong Buy)"
        elif decision_counts["Moderate"] >= decision_counts["Best"] and decision_counts["Moderate"] >= decision_counts["Bad"]:
            classification = "Moderate (Hold)"
        else:
            classification = "Bad (Avoid)"

        # Add classification and last update to the data
        data["Classification"] = classification
        data["Last Update (Most Recent Quarter)"] = last_update

        return data
    except Exception as e:
        return {"Error": str(e)}

# Sector-wise News Sentiment Analysis
def fetch_sector_sentiment(sector, nifty_50_tickers):
    sector_tickers = [ticker for ticker, sec in nifty_50_tickers.items() if sec == sector]
    sentiment_data = []
    
    for ticker in sector_tickers:
        news_data = get_news_data(ticker)
        if news_data is not None and not news_data.empty:
            news_data = analyze_sentiment(news_data)
            sentiment_counts = news_data['sentiment'].value_counts()
            sentiment_data.append({
                "Ticker": ticker,
                "Positive": sentiment_counts.get('positive', 0),
                "Neutral": sentiment_counts.get('neutral', 0),
                "Negative": sentiment_counts.get('negative', 0)
            })
    
    return pd.DataFrame(sentiment_data)

# --- Dummy functions for demonstration ---
def get_model_predictions():
    # Make predictions perfectly match true labels for demonstration
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()  # All predictions are correct
    return y_true, y_pred

def get_loss_accuracy_history():
    # Replace with your training history
    epochs = np.arange(1, 21)
    test_acc = np.random.uniform(0.7, 1.0, 20)
    test_loss = np.random.uniform(0.2, 0.5, 20)
    val_acc = np.random.uniform(0.7, 1.0, 20)
    val_loss = np.random.uniform(0.2, 0.5, 20)
    return epochs, test_acc, test_loss, val_acc, val_loss

def get_polarity_scores():
    # Replace with your real-time polarity scores
    times = pd.date_range(end=datetime.now(), periods=20)
    polarity = np.random.uniform(-1, 1, 20)
    return times, polarity

# --- Add to sidebar ---
def analytics_sidebar():
    st.sidebar.header("Analytical Metrics")
    show_confusion = st.sidebar.checkbox("Confusion Matrix", True)
    show_f1 = st.sidebar.checkbox("F1 Score", True)
    show_test_acc_loss = st.sidebar.checkbox("Test Accuracy & Loss", True)
    show_val_acc_loss = st.sidebar.checkbox("Validation Accuracy & Loss", True)
    show_polarity = st.sidebar.checkbox("Polarity Score", True)
    return {
        "confusion": show_confusion,
        "f1": show_f1,
        "test_acc_loss": show_test_acc_loss,
        "val_acc_loss": show_val_acc_loss,
        "polarity": show_polarity
    }

# --- Main analytics section ---
def analytics_section(selected_metrics, api_url="http://localhost:8000/api"):
    st.header("📊 Analytical Report")
    y_true, y_pred = fetch_realtime_model_predictions(api_url)
    epochs, test_acc, test_loss, val_acc, val_loss = fetch_realtime_loss_accuracy(api_url)
    times, polarity = fetch_realtime_polarity_scores(api_url)

    if selected_metrics["confusion"]:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)

    if selected_metrics["f1"]:
        st.subheader("F1 Score")
        f1 = f1_score(y_true, y_pred)
        st.metric("F1 Score", f"{f1:.2f}")

        # F1 Score Over Epochs Graph (API)
        response = requests.get("http://localhost:8000/api/f1_score")
        if response.status_code == 200:
            data = response.json()
            st.subheader("F1 Score Over Epochs")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["epochs"],
                y=data["f1_score"],
                mode='lines+markers',
                name='F1 Score'
            ))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="F1 Score",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to fetch F1 score data.")

    if selected_metrics["test_acc_loss"]:
        st.subheader("Test Accuracy & Loss")
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=epochs, y=test_acc, mode='lines+markers', name='Test Accuracy'))
        fig_test.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines+markers', name='Test Loss'))
        fig_test.update_layout(xaxis_title="Epoch", yaxis_title="Value", template="plotly_dark")
        st.plotly_chart(fig_test, use_container_width=True)

    if selected_metrics["val_acc_loss"]:
        st.subheader("Validation Accuracy & Loss")
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))
        fig_val.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))
        fig_val.update_layout(xaxis_title="Epoch", yaxis_title="Value", template="plotly_dark")
        st.plotly_chart(fig_val, use_container_width=True)

    if selected_metrics["polarity"]:
        st.subheader("Polarity Score Over Time")
        fig_polarity = go.Figure()
        fig_polarity.add_trace(go.Scatter(x=times, y=polarity, mode='lines+markers', name='Polarity Score'))
        fig_polarity.update_layout(xaxis_title="Time", yaxis_title="Polarity", template="plotly_dark")
        st.plotly_chart(fig_polarity, use_container_width=True)

# Fetch real-time model predictions
def fetch_realtime_model_predictions(api_url):
    response = requests.get(f"{api_url}/predictions")
    data = response.json()
    return data["y_true"], data["y_pred"]

# Fetch real-time loss and accuracy metrics
def fetch_realtime_loss_accuracy(api_url):
    response = requests.get(f"{api_url}/metrics")
    data = response.json()
    return data["epochs"], data["test_acc"], data["test_loss"], data["val_acc"], data["val_loss"]

# Fetch real-time polarity scores
def fetch_realtime_polarity_scores(api_url):
    response = requests.get(f"{api_url}/polarity")
    data = response.json()
    return data["times"], data["polarity"]

# Main function
def main():
    os.environ["STREAMLIT_WATCH_FILE"] = "false"
    
    st.title("📈 Indian Stock Market Technical & News Sentiment Analysis")
    
    inputs = sidebar()
    
    # Check if "All" is selected for ticker and sector
    if inputs["ticker"] == "All" and inputs["sector"] == "All":
        st.warning("Please select a specific stock or sector to view the analysis.")
        return  # Stop execution if "All" is selected for both
    
    # Set black theme
    plotly_template = "plotly_dark"
    
    # Handle case where sector is selected but stock is "All"
    if inputs["ticker"] == "All" and inputs["sector"] != "All":
        st.subheader(f"📰 {inputs['sector']} Sector News Sentiment Analysis")
        with st.spinner("Fetching sector-wise sentiment data..."):
            nifty_50_tickers = {
                "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "INFY.NS": "IT", "ICICIBANK.NS": "Banking",
                "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "KOTAKBANK.NS": "Banking", "SBIN.NS": "Banking", "BHARTIARTL.NS": "Telecom",
                "ADANIENT.NS": "Infrastructure", "LT.NS": "Infrastructure", "AXISBANK.NS": "Banking", "ASIANPAINT.NS": "Paints", "BAJFINANCE.NS": "Finance",
                "HCLTECH.NS": "IT", "MARUTI.NS": "Automobile", "SUNPHARMA.NS": "Pharma", "TITAN.NS": "Jewelry", "ULTRACEMCO.NS": "Cement",
                "WIPRO.NS": "IT", "ONGC.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy", "COALINDIA.NS": "Energy",
                "JSWSTEEL.NS": "Steel", "TATAMOTORS.NS": "Automobile", "ADANIPORTS.NS": "Infrastructure", "GRASIM.NS": "Cement", "BPCL.NS": "Energy",
                "INDUSINDBK.NS": "Banking", "DIVISLAB.NS": "Pharma", "EICHERMOT.NS": "Automobile", "HEROMOTOCO.NS": "Automobile", "DRREDDY.NS": "Pharma",
                "BRITANNIA.NS": "FMCG", "APOLLOHOSP.NS": "Healthcare", "CIPLA.NS": "Pharma", "BAJAJFINSV.NS": "Finance", "HDFCLIFE.NS": "Insurance",
                "SBILIFE.NS": "Insurance", "TECHM.NS": "IT", "TATACONSUM.NS": "FMCG", "M&M.NS": "Automobile", "SHREECEM.NS": "Cement",
                "UPL.NS": "Agrochemicals", "ICICIPRULI.NS": "Insurance", "PIDILITIND.NS": "Chemicals", "DLF.NS": "Real Estate", "BAJAJ-AUTO.NS": "Automobile"
            }
            sector_sentiment = fetch_sector_sentiment(inputs["sector"], nifty_50_tickers)
        
        if not sector_sentiment.empty:
            # Aggregate sentiment counts for the sector
            total_positive = sector_sentiment["Positive"].sum()
            total_neutral = sector_sentiment["Neutral"].sum()
            total_negative = sector_sentiment["Negative"].sum()

            # Display sentiment distribution as a pie chart
            sentiment_fig = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[total_positive, total_neutral, total_negative],
                hole=0.3,
                marker_colors=["green", "gray", "red"]
            ))
            sentiment_fig.update_layout(
                title=f"{inputs['sector']} Sector Sentiment Distribution",
                template="plotly_dark"
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)

            # Display the sentiment data table
            st.table(sector_sentiment)
        else:
            st.error("No sentiment data available for the selected sector.")

        # Skip further analysis since no specific stock is selected
        st.info("Please select a specific stock to view the price chart and fundamental analysis.")
        return

    # Date range calculation
    end_date = datetime.now()
    if inputs["analysis_period"] == "1M":
        start_date = end_date - timedelta(days=60)  # Fetch 2 months of data for 1M period
    elif inputs["analysis_period"] == "3M":
        start_date = end_date - timedelta(days=90)
    elif inputs["analysis_period"] == "6M":
        start_date = end_date - timedelta(days=180)
    elif inputs["analysis_period"] == "1Y":
        start_date = end_date - timedelta(days=365)
    else:  # YTD
        start_date = datetime(end_date.year, 1, 1)
    
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        stock_data = get_stock_data(
            inputs["ticker"],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        if stock_data is None or stock_data.empty:
            st.error("No stock data available for the selected period.")
            st.stop()

    # Convert start_date and end_date to timezone-aware
    tz = timezone("Asia/Kolkata")  # Use the same timezone as stock_data.index
    start_date = tz.localize(start_date)
    end_date = tz.localize(end_date)

    # Filter data for the selected period
    filtered_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
    filtered_data = calculate_technical_indicators(filtered_data)

    # Plot price chart and indicators
    # Use filtered_data for all visualizations
    
    # Price chart
    st.subheader(f"{inputs['ticker']} Price Chart")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # Adjust spacing between subplots
        row_heights=[0.7, 0.3]  # Adjust the height ratio of the two subplots
    )

    # Candlestick for price (row 1)
    fig.add_trace(go.Candlestick(
        x=filtered_data.index,
        open=filtered_data['Open'],
        high=filtered_data['High'],
        low=filtered_data['Low'],
        close=filtered_data['Close'],
        name="Price"
    ), row=1, col=1)

    # Plot SMAs (row 1)
    if inputs["show_sma_9"] and 'sma_9' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['sma_9'],
            line=dict(color='purple', width=1),
            name="SMA 9"
        ), row=1, col=1)

    if inputs["show_sma_20"] and 'sma_20' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['sma_20'],
            line=dict(color='blue', width=1),
            name="SMA 20"
        ), row=1, col=1)

    if inputs["show_sma_50"] and 'sma_50' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['sma_50'],
            line=dict(color='green', width=1),
            name="SMA 50"
        ))

    if inputs["show_sma_100"] and 'sma_100' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['sma_100'],
            line=dict(color='orange', width=1),
            name="SMA 100"
        ))

    if inputs["show_sma_200"] and 'sma_200' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['sma_200'],
            line=dict(color='red', width=1),
            name="SMA 200"
        ))

    # Exponential Moving Average
    if inputs["show_ema"] and 'ema_20' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['ema_20'],
            line=dict(color='orange', width=1),
            name="EMA 20"
        ))

    # Bollinger Bands
    if inputs["show_bollinger"] and all(col in filtered_data.columns for col in ['bb_upper', 'bb_lower']):
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['bb_upper'],
            line=dict(color='gray', width=1),
            name="Upper Bollinger Band"
        ))
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['bb_lower'],
            line=dict(color='gray', width=1),
            name="Lower Bollinger Band",
            fill='tonexty'
        ))

    # Add volume as a bar chart (row 2)
    fig.add_trace(go.Bar(
        x=filtered_data.index,
        y=filtered_data['Volume'],
        name="Volume",
        marker=dict(color='rgba(0, 128, 255, 0.5)')
    ), row=2, col=1)

    # Update layout for the subplots
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template=plotly_template,
        height=700,  # Total height of the chart
        yaxis=dict(title="Price"),  # Y-axis for the price chart
        yaxis2=dict(title="Volume")  # Y-axis for the volume chart
    )

    # Add last price annotation (row 1)
    last_price = filtered_data['Close'].iloc[-1]
    last_date = filtered_data.index[-1]
    fig.add_annotation(
        x=last_date,
        y=last_price,
        text=f"Last Price: {float(last_price):.2f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="white"),
        bgcolor="black",
        bordercolor="white",
        row=1, col=1
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the last current price
    st.write(f"**Last Current Price:** ₹{float(last_price):.2f}")

    # Technical indicator charts
    col1, col2 = st.columns(2)
    
    with col1:
        if inputs["show_rsi"] and 'rsi' in filtered_data.columns:
            st.subheader("RSI (14-day)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['rsi'],
                line=dict(color='purple', width=2),
                name="RSI"
            ))
            rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
            rsi_fig.update_layout(template=plotly_template, height=300)
            st.plotly_chart(rsi_fig, use_container_width=True)
    
    with col2:
        if inputs["show_macd"] and all(col in filtered_data.columns for col in ['macd', 'macd_signal']):
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['macd'],
                line=dict(color='blue', width=2),
                name="MACD"
            ))
            macd_fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['macd_signal'],
                line=dict(color='orange', width=2),
                name="Signal"
            ))
            macd_fig.update_layout(template=plotly_template, height=300)
            st.plotly_chart(macd_fig, use_container_width=True)

    # Fundamental Analysis Section (Optional)
    if inputs["show_fundamental"]:
        st.subheader("📊 Fundamental Analysis")
        with st.spinner("Fetching fundamental data..."):
            fundamental_data = fetch_fundamental_data(inputs["ticker"])

        if fundamental_data and "Error" not in fundamental_data:
            # Convert all values to strings to avoid mixed types and include decisions
            decisions = {
                "EPS": "Best" if fundamental_data.get("EPS", 0) > 20 else "Moderate" if 10 <= fundamental_data.get("EPS", 0) <= 20 else "Bad",
                "P/E": "Best" if fundamental_data.get("P/E", 0) < 15 else "Moderate" if 15 <= fundamental_data.get("P/E", 0) <= 25 else "Bad",
                "P/B": "Best" if fundamental_data.get("P/B", 0) < 3 else "Moderate" if 3 <= fundamental_data.get("P/B", 0) <= 5 else "Bad",
                "D/E": "Best" if fundamental_data.get("D/E", 0) < 0.5 else "Moderate" if 0.5 <= fundamental_data.get("D/E", 0) <= 1.5 else "Bad",
                "PEG": "Best" if fundamental_data.get("PEG", 0) < 1.5 else "Moderate" if 1.5 <= fundamental_data.get("PEG", 0) <= 2 else "Bad",
                "EV/EBITDA": "Best" if fundamental_data.get("EV/EBITDA", 0) < 8 else "Moderate" if 8 <= fundamental_data.get("EV/EBITDA", 0) <= 10 else "Bad",
                "Revenue Growth (%)": "Best" if fundamental_data.get("Revenue Growth (%)", 0) > 15 else "Moderate" if 10 <= fundamental_data.get("Revenue Growth (%)", 0) <= 15 else "Bad",
                "Net Profit Margin (%)": "Best" if fundamental_data.get("Net Profit Margin (%)", 0) > 20 else "Moderate" if 10 <= fundamental_data.get("Net Profit Margin (%)", 0) <= 20 else "Bad",
                "ROE (%)": "Best" if fundamental_data.get("ROE (%)", 0) > 15 else "Moderate" if 10 <= fundamental_data.get("ROE (%)", 0) <= 15 else "Bad",
                "ROCE (%)": "Best" if fundamental_data.get("ROCE (%)", 0) > 15 else "Moderate" if 10 <= fundamental_data.get("ROCE (%)", 0) <= 15 else "Bad",
                "FCF Trend": "Best" if fundamental_data.get("FCF Trend", "") == "Positive & Growing" else "Bad",
                "OCF Trend": "Best" if fundamental_data.get("OCF Trend", "") == "Positive & Growing" else "Bad",
            }

            fundamental_df = pd.DataFrame(
                [
                    (key, f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value), decisions.get(key, "N/A"))
                    for key, value in fundamental_data.items()
                    if key != "Classification" and key != "Last Update (Most Recent Quarter)"
                ],
                columns=["Metric", "Value", "Decision"]
            )
            # Add classification and last update as separate rows
            classification_row = pd.DataFrame([["Classification", fundamental_data["Classification"], ""]], columns=["Metric", "Value", "Decision"])
            last_update_row = pd.DataFrame([["Last Update (Most Recent Quarter)", fundamental_data["Last Update (Most Recent Quarter)"], ""]], columns=["Metric", "Value", "Decision"])
            fundamental_df = pd.concat([fundamental_df, classification_row, last_update_row], ignore_index=True)

            st.table(fundamental_df)
        else:
            st.error(f"Error fetching data: {fundamental_data.get('Error', 'Unknown error')}")

    # Sector-wise News Sentiment Analysis with Pie Chart
    if inputs["sector"] != "All":
        st.subheader(f"📰 {inputs['sector']} Sector News Sentiment Analysis")
        with st.spinner("Fetching sector-wise sentiment data..."):
            nifty_50_tickers = {
                "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "INFY.NS": "IT", "ICICIBANK.NS": "Banking",
                "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "KOTAKBANK.NS": "Banking", "SBIN.NS": "Banking", "BHARTIARTL.NS": "Telecom",
                "ADANIENT.NS": "Infrastructure", "LT.NS": "Infrastructure", "AXISBANK.NS": "Banking", "ASIANPAINT.NS": "Paints", "BAJFINANCE.NS": "Finance",
                "HCLTECH.NS": "IT", "MARUTI.NS": "Automobile", "SUNPHARMA.NS": "Pharma", "TITAN.NS": "Jewelry", "ULTRACEMCO.NS": "Cement",
                "WIPRO.NS": "IT", "ONGC.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy", "COALINDIA.NS": "Energy",
                "JSWSTEEL.NS": "Steel", "TATAMOTORS.NS": "Automobile", "ADANIPORTS.NS": "Infrastructure", "GRASIM.NS": "Cement", "BPCL.NS": "Energy",
                "INDUSINDBK.NS": "Banking", "DIVISLAB.NS": "Pharma", "EICHERMOT.NS": "Automobile", "HEROMOTOCO.NS": "Automobile", "DRREDDY.NS": "Pharma",
                "BRITANNIA.NS": "FMCG", "APOLLOHOSP.NS": "Healthcare", "CIPLA.NS": "Pharma", "BAJAJFINSV.NS": "Finance", "HDFCLIFE.NS": "Insurance",
                "SBILIFE.NS": "Insurance", "TECHM.NS": "IT", "TATACONSUM.NS": "FMCG", "M&M.NS": "Automobile", "SHREECEM.NS": "Cement",
                "UPL.NS": "Agrochemicals", "ICICIPRULI.NS": "Insurance", "PIDILITIND.NS": "Chemicals", "DLF.NS": "Real Estate", "BAJAJ-AUTO.NS": "Automobile"
            }
            sector_sentiment = fetch_sector_sentiment(inputs["sector"], nifty_50_tickers)
        
        if not sector_sentiment.empty:
            # Aggregate sentiment counts for the sector
            total_positive = sector_sentiment["Positive"].sum()
            total_neutral = sector_sentiment["Neutral"].sum()
            total_negative = sector_sentiment["Negative"].sum()

            # Display sentiment distribution as a pie chart
            sentiment_fig = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[total_positive, total_neutral, total_negative],
                hole=0.3,
                marker_colors=["green", "gray", "red"]
            ))
            sentiment_fig.update_layout(
                title=f"{inputs['sector']} Sector Sentiment Distribution",
                template="plotly_dark"
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)

            # Display the sentiment data table
            st.table(sector_sentiment)
        else:
            st.error("No sentiment data available for the selected sector.")

    # News sentiment analysis
    st.subheader("📰 Latest News Sentiment Analysis")
    
    with st.spinner("Fetching and analyzing news..."):
        news_data = get_news_data(inputs["ticker"])
    
    if news_data is not None and not news_data.empty:
        news_data = analyze_sentiment(news_data)
        
        # Sentiment summary
        sentiment_counts = news_data['sentiment'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive News", sentiment_counts.get('positive', 0))
        with col2:
            st.metric("Neutral News", sentiment_counts.get('neutral', 0))
        with col3:
            st.metric("Negative News", sentiment_counts.get('negative', 0))
        
        # Sentiment pie chart
        sentiment_fig = go.Figure(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=['green', 'gray', 'red']
        ))
        sentiment_fig.update_layout(
            title="Sentiment Distribution",
            template=plotly_template
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # News articles
        st.subheader("Latest News Articles")
        for _, row in news_data.iterrows():
            with st.expander(f"{row['title']} ({row['sentiment']})"):
                st.write(f"**Source:** {row['source']} | **Published At:** {row['publishedAt']}")
                st.write(row['description'])
                st.write(f"[Read more]({row['url']})")
                st.write(f"**Sentiment Score:** {row['sentiment_score']:.2f}")

    analytics_metrics = analytics_sidebar()
    st.markdown("---")
    analytics_section(analytics_metrics)

if __name__ == "__main__":
    main()