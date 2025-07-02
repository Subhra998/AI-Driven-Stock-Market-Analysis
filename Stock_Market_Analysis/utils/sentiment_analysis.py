from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import numpy as np
import pandas as pd

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
# transformer_analyzer = pipeline("sentiment-analysis")  # Uncomment for transformer-based analysis

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    if not text or pd.isna(text):
        return 0
    return vader_analyzer.polarity_scores(text)['compound']

def analyze_sentiment(df):
    """Perform sentiment analysis on news data"""
    if df.empty:
        return df
    
    # Analyze sentiment for title and description
    df['title_sentiment'] = df['title'].apply(analyze_sentiment_vader)
    df['desc_sentiment'] = df['description'].apply(analyze_sentiment_vader)
    
    # Combine scores (weight title more heavily)
    df['sentiment_score'] = 0.7 * df['title_sentiment'] + 0.3 * df['desc_sentiment']
    
    # Classify sentiment
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )
    
    return df