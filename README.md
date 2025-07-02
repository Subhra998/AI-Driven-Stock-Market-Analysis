# 📈 Indian Stock Market Technical & News Sentiment Analysis

This project is a real-time analytics dashboard for the Indian stock market, combining technical analysis, news sentiment analysis, and machine learning model evaluation. It provides interactive visualizations, sector-wise and stock-specific insights, and key model performance metrics.

---

## Features

- **Stock Price Visualization:** Candlestick charts with technical indicators (SMA, EMA, Bollinger Bands, RSI, MACD, etc.)
- **News Sentiment Analysis:** Real-time news fetching and sentiment classification for selected stocks and sectors.
- **Fundamental Analysis:** Key financial metrics and qualitative decisions for Nifty 50 stocks.
- **Model Evaluation:** Confusion matrix, F1 score, accuracy, precision, recall, and loss/accuracy plots over epochs.
- **Sector-wise Analysis:** Aggregate sentiment and performance metrics for different sectors.
- **Interactive Dashboard:** Built with Streamlit and Plotly for easy exploration.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [scikit-learn](https://scikit-learn.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [requests](https://docs.python-requests.org/)
- [FastAPI](https://fastapi.tiangolo.com/) (for backend API)
- [Uvicorn](https://www.uvicorn.org/) (for running FastAPI server)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### Running the Project

1. **Start the Backend API:**
   ```bash
   uvicorn api_server:app --reload
   ```

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501` to use the dashboard.

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard (main frontend)
├── api_server.py           # FastAPI backend for model predictions and metrics
├── utils/                  # Utility modules (data fetching, analysis, etc.)
│   ├── data_fetcher.py
│   ├── sentiment_analysis.py
│   └── technical_analysis.py
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Screenshots

*Add screenshots of your dashboard and analytics here for better presentation.*

---

## Customization

- To add more stocks or sectors, update the `nifty_50_tickers` dictionary in `app.py`.
- To improve model performance, update the backend logic in `api_server.py` with your own ML models and datasets.
- For a white background, edit the CSS in `load_css()` or use Streamlit’s theme settings.

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

*For any questions or contributions, please open an issue or pull request!*
