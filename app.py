import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. CONFIGURATIE ---
API_KEY = "d5h3vm9r01qll3dlm2sgd5h3vm9r01qll3dlm2t0"
st.set_page_config(page_title="AI Finnhub Dashboard", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT"]

# --- 2. FINNHUB API FUNCTIES ---
def get_finnhub_data(ticker):
    # Basis koers data
    quote = requests.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={API_KEY}").json()
    
    # Kaars data (laatste 100 dagen)
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=150)).timestamp())
    candles = requests.get(f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start}&to={end}&token={API_KEY}").json()
    
    # Earnings data
    earnings = requests.get(f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={API_KEY}").json()
    
    # News Sentiment
    sentiment = requests.get(f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={API_KEY}").json()
    
    return quote, candles, earnings, sentiment

# --- 3. AI MODEL (Ensemble) ---
def run_ai_logic(candles):
    if 'c' not in candles: return None, None
    df = pd.DataFrame({'Close': candles['c']})
    df['Target'] = df['Close'].shift(-1)
    train = df.dropna()
    
    X = train[['Close']].values
    y = train['Target'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    current_price = df['Close'].iloc[-1]
    prediction = model.predict([[current_price]])[0]
    return prediction, current_price

# --- 4. DASHBOARD UI ---
st.title("🚀 Finnhub AI Trading Terminal")

with st.sidebar:
    st.header("⭐ Watchlist")
    new_stock = st.text_input("Voeg USA Ticker toe:").upper()
    if st.button("Toevoegen") and new_stock:
        st.session_state.watchlist.append(new_stock)
        st.rerun()
    
    selected_stock = st.selectbox("Selecteer aandeel:", st.session_state.watchlist)

if selected_stock:
    with st.spinner('Finnhub data ophalen...'):
        quote, candles, earnings, sentiment = get_finnhub_data(selected_stock)
        
        if 'c' in candles:
            pred_price, current_price = run_ai_logic(candles)
            
            # Sentiment berekening
            sent_val = sentiment.get('sentiment', {}).get('bullishPercent', 0.5) if sentiment else 0.5
            
            # AI Score (Prijsactie + Finnhub Sentiment)
            price_move = (pred_price - current_price) / current_price
            ai_score = (50 + (price_move * 500)) * 0.7 + (sent_val * 100) * 0.3
            ai_score = min(max(ai_score, 0), 100)

            # Statistieken
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prijs", f"${current_price:.2f}")
            
            # Earnings Datum
            next_earn = earnings[0]['period'] if earnings else "N/A"
            c2.metric("Earnings Periode", next_earn)
            
            # Sentiment Gauge
            c3.metric("Bullish Sentiment", f"{sent_val*100:.0f}%")
            c4.metric("Totaal AI Score", f"{ai_score:.1f}/100")

            # --- VISUALISATIE ---
            st.subheader("Marktanalyse & Sentiment")
            col_left, col_right = st.columns([2, 1])

            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(candles['c']))), y=candles['c'], name="Koers", line=dict(color='#00ff88')))
                fig.update_layout(title=f"Koersverloop {selected_stock}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Sentiment Meter (Gauge Chart)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = ai_score,
                    title = {'text': "AI Rating"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "gold"},
                        'steps': [
                            {'range': [0, 40], 'color': "red"},
                            {'range': [40, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "green"}]}))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Handels Signaal
            if ai_score > 75:
                st.success(f"### 📈 STERK KOOP SIGNAAL: De AI voorziet een stijging naar ${pred_price:.2f}")
            elif ai_score < 35:
                st.error(f"### 📉 VERKOOP SIGNAAL: Finnhub data wijst op neerwaartse druk.")
            else:
                st.info("### ⚖️ NEUTRAAL: Geen duidelijke trend gedetecteerd.")

        else:
            st.error("Kon geen data ophalen. Controleer of de API-key nog geldig is of de limiet bereikt is.")


