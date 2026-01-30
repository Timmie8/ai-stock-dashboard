import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. CONFIGURATIE ---
API_KEY = "d5h3vm9r01qll3dlm2sgd5h3vm9r01qll3dlm2t0"
st.set_page_config(page_title="AI Finnhub Terminal", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT"]

# --- 2. FINNHUB API FUNCTIE (Met extra veiligheid) ---
def get_finnhub_data(ticker):
    base_url = "https://finnhub.io/api/v1"
    
    # 1. Quote (Huidige prijs)
    q = requests.get(f"{base_url}/quote?symbol={ticker}&token={API_KEY}").json()
    
    # 2. Candles (Historie voor AI)
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=100)).timestamp())
    c = requests.get(f"{base_url}/stock/candle?symbol={ticker}&resolution=D&from={start}&to={end}&token={API_KEY}").json()
    
    # 3. Sentiment (Optioneel, Finnhub geeft dit niet voor alle kleine tickers)
    s = requests.get(f"{base_url}/news-sentiment?symbol={ticker}&token={API_KEY}").json()
    
    # 4. Basic Financials (voor Earnings datum)
    f = requests.get(f"{base_url}/stock/metric?symbol={ticker}&metric=all&token={API_KEY}").json()
    
    return q, c, s, f

# --- 3. DASHBOARD LOGICA ---
st.title("📈 AI Trading Terminal (Finnhub Powered)")

with st.sidebar:
    st.header("⭐ Watchlist")
    new_stock = st.text_input("Voeg Ticker toe (USA):").upper()
    if st.button("Toevoegen") and new_stock:
        st.session_state.watchlist.append(new_stock)
        st.rerun()
    selected_stock = st.selectbox("Analyseer:", st.session_state.watchlist)

if selected_stock:
    q, c, s, f = get_finnhub_data(selected_stock)
    
    # Check of we basisdata hebben
    if q and 'c' in q and q['c'] != 0:
        current_price = q['c']
        
        # Sentiment veiligstellen
        # In het gratis plan is 'bullishPercent' soms 'None', we zetten het dan op 50%
        try:
            bullish_pct = s['sentiment']['bullishPercent'] if s and 'sentiment' in s else 0.5
            if bullish_pct is None: bullish_pct = 0.5
        except:
            bullish_pct = 0.5

        # AI Voorspelling (Simpel Ensemble op basis van recente kaarsen)
        if 'c' in c:
            df = pd.DataFrame({'price': c['c']})
            df['next'] = df['price'].shift(-1)
            X = df[['price']].iloc[:-1]
            y = df['next'].iloc[:-1]
            model = RandomForestRegressor(n_estimators=50).fit(X, y)
            pred_price = model.predict([[current_price]])[0]
        else:
            pred_price = current_price

        # Score berekening
        ai_score = (bullish_pct * 100 * 0.4) + (50 + (pred_price - current_price) / current_price * 1000) * 0.6
        ai_score = min(max(ai_score, 0), 100)

        # UI: Metrieken
        col1, col2, col3 = st.columns(3)
        col1.metric("Huidige Koers", f"${current_price:.2f}")
        
        # Earnings datum uit de financials halen
        earn_date = f.get('metric', {}).get('earningsReleaseDate', 'Binnenkort')
        col2.metric("Earnings Verwacht", earn_date)
        col3.metric("AI Score", f"{ai_score:.1f}/100")

        # Visualisatie
        st.subheader("Analyse & Signaal")
        left, right = st.columns([2, 1])
        
        with left:
            if 'c' in c:
                fig = go.Figure(data=[go.Candlestick(x=list(range(len(c['c']))),
                                open=c['o'], high=c['h'], low=c['l'], close=c['c'])])
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with right:
            st.write("### AI Rating")
            if ai_score >= 70: st.success("🚀 STERK KOOP")
            elif ai_score <= 30: st.error("⚠️ VERKOOP")
            else: st.info("⚖️ NEUTRAAL")
            
            st.write(f"**Bullish Sentiment:** {bullish_pct*100:.1f}%")
            st.write(f"**AI Koersdoel:** ${pred_price:.2f}")

    else:
        st.warning(f"Geen live data voor {selected_stock}. Dit kan gebeuren als de beurs gesloten is of de ticker incorrect is.")



