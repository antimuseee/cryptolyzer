# CoinGecko API Key Setup

## 🚀 Enhanced Features with API Key

With a CoinGecko API key, your Cryptolyzer app will have:
- ✅ **100 coins analyzed** (vs 25 without API key)
- ✅ **Faster analysis** (30 seconds vs 60 seconds)
- ✅ **Higher rate limits** (no more rate limit errors)
- ✅ **More reliable** data fetching

## 🔑 How to Get Your API Key

1. Visit: https://www.coingecko.com/en/api/pricing
2. Choose a plan (Demo API is free)
3. Sign up and get your API key

## ⚙️ Setting Up the API Key

### For Local Development:
Create a `.env` file in your project root:
```
COINGECKO_API_KEY=your_actual_api_key_here
```

### For Render Deployment:
1. Go to your Render dashboard
2. Select your Cryptolyzer app
3. Go to "Environment" tab
4. Add environment variable:
   - **Key:** `COINGECKO_API_KEY`
   - **Value:** `your_actual_api_key_here`
5. Save and redeploy

## 🎯 Benefits

- **No more rate limit errors**
- **Faster analysis times**
- **More comprehensive coin coverage**
- **Better user experience**

## 🔒 Security Note

Never commit your API key to Git. Always use environment variables! 