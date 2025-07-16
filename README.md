# Crypto Volatility Analyzer

A web application that identifies cryptocurrency coins with the most fluctuations in value while maintaining low volatility. This tool helps investors find potentially profitable trading opportunities by analyzing historical price data.

## Features

- Analyzes top 50 cryptocurrencies by market cap
- Calculates volatility (standard deviation of price changes)
- Calculates price change over a specified period
- Sorts coins by volatility and price change
- Displays results in an easy-to-read table
- Real-time data from CoinGecko API

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python crypto_volatility.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Analyze Coins" button to fetch and analyze the data
2. The application will display the top 10 coins with the lowest volatility and highest price change
3. Each coin's information includes:
   - Name and symbol
   - Current price in USD
   - Volatility percentage
   - Price change percentage

## Technical Details

- Built with Flask (Python web framework)
- Uses CoinGecko API for cryptocurrency data
- Frontend built with Bootstrap and vanilla JavaScript
- Data analysis performed using pandas
