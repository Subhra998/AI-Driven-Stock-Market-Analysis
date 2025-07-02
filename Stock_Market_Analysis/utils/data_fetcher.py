import time
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob  # Install with `pip install textblob`

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for the given ticker and date range.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def get_news_data(ticker, max_pages=1):  # Limit to 1 page for free tier
    """
    Fetch news articles for the given ticker with pagination and perform sentiment analysis.
    """
    api_key = "b4811a1a4a644baaabc3fbdcf4a1c746"
    base_url = "https://newsapi.org/v2/everything"
    query = ticker.split('.')[0]  # Use the stock name without the ".NS" suffix
    all_articles = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": 100,  # Maximum allowed by NewsAPI
            "page": page,
            "sortBy": "publishedAt",
            "language": "en"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if not articles:
                break  # Stop if no more articles are available
            all_articles.extend(articles)
        elif response.status_code == 426 or response.status_code == 400:  # Handle free tier limit
            print(f"Warning: API limit reached. Only the first 100 results are available.")
            break
        else:
            print(f"Error fetching news: {response.status_code}")
            print(f"Response content: {response.text}")
            break

    # Convert to DataFrame
    news_df = pd.DataFrame(all_articles)
    if not news_df.empty:
        news_df = news_df.rename(columns={
            "title": "title",
            "source": "source",
            "publishedAt": "publishedAt",
            "description": "description",
            "url": "url"
        })
        news_df["source"] = news_df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)

        # Perform sentiment analysis
        def analyze_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

        news_df["sentiment"] = news_df["description"].apply(lambda x: analyze_sentiment(x) if isinstance(x, str) else 0)

        # Filter for American and Indian stock market sentiment
        keywords = ["NASDAQ", "Dow Jones", "S&P 500", "NYSE", "BSE", "NSE", "Sensex", "Nifty"]
        news_df = news_df[news_df["title"].str.contains('|'.join(keywords), case=False, na=False)]

    return news_df