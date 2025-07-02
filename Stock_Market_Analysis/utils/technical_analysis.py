def calculate_technical_indicators(data):
    """Calculate various technical indicators."""
    if data is None or data.empty or 'Close' not in data.columns:
        return data

    if len(data) < 26:  # Minimum data points for MACD
        print("Not enough data to calculate indicators.")
        return data

    # Create a copy of the DataFrame to avoid modifying the original slice
    data = data.copy()

    # Calculate SMAs
    data['sma_9'] = data['Close'].rolling(window=9).mean()
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['sma_100'] = data['Close'].rolling(window=100).mean()
    data['sma_200'] = data['Close'].rolling(window=200).mean()

    # Calculate EMA
    data['ema_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Calculate Bollinger Bands
    data['bb_upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['bb_lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    return data