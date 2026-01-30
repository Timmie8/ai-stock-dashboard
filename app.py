import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# 1. INITIALISATIE (Dit moet bovenaan!)
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "ASML.AS"]

try:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except:
    pass

# 2. FUNCTIES
def get_data_and_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        news = stock.news
        calendar = stock.calendar
        next_earn = "Onbekend"
        if calendar is not None and not calendar.empty:
            next_earn = calendar.iloc[0, 0].strftime('%d-%m-%Y')
        return df, news, next_earn
    except:
        return pd.DataFrame(), [], "Fout"

def analyze_sentiment(news_list):
    if not news_list: return 0
    scores = [sia.polarity_scores(n['title'])['compound'] for n in news_list[:5]]
    return np.mean(scores)

def train_ai_models(df):
    if df.empty: return 0, 0
    df_clean = df[['Close']].copy()
    df_clean['Target'] = df_clean['Close'].shift(-1)
    df_clean.dropna(inplace=True)
    rf = RandomForestRegressor(n_estimators=50).fit(df_clean[['Close']][:-1], df_clean['Target'][:-1])
    last_price = df_clean['Close'].iloc[-1]
    pred_rf = rf.predict(np.array([[last_price]]))[0]
    return pred_rf, last_price

# 3. UI LAYOUT
st.title("🤖 AI Stock Intelligence Dashboard")

with st.sidebar:
    st.header("⭐ Watchlist")
    new_ticker = st.text_input("Voeg Ticker toe:").upper()
    if st.button("Toevoegen") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.rerun()
    
    selected_stock = st.selectbox("Kies een aandeel:", st.session_state.watchlist)
    
    st.markdown("---")
    if st.button("🗑️ Wis Watchlist"):
        st.session_state.watchlist = ["AAPL"]
        st.rerun()

# 4. HOOFDSCHERM LOGICA
if selected_stock:
    df, news, earnings_date = get_data_and_news(selected_stock)
    
    if not df.empty:
        pred_price, current_price = train_ai_models(df)
        sent_score = analyze_sentiment(news)
        
        # AI Score Logica
        price_diff = (pred_price - current_price) / current_price
        final_score = min(max(50 + (price_diff * 500) + (sent_score * 20), 0), 100)

        # Statistieken
        c1, c2, c3 = st.columns(3)
        c1.metric("Huidige Koers", f"${current_price:.2f}")
        c2.metric("Earnings Datum", earnings_date)
        c3.metric("AI Vertrouwen", f"{final_score:.1f}/100")

        # Signaal
        if final_score >= 75: st.success("🚀 STERK KOOP SIGNAAL")
        elif final_score <= 25: st.error("⚠️ VERKOOP SIGNAAL")
        else: st.info("⚖️ NEUTRAAL HOUDEN")

        # Grafiek
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'][-60:], name="Koers"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Kon geen data ophalen voor dit aandeel.")
else:
    st.info("Selecteer een aandeel in de sidebar om te beginnen.")

# 5. RAPPORTAGE (Onderaan om fouten te voorkomen)
st.sidebar.markdown("---")
if st.sidebar.button("📊 Genereer Rapport"):
    st.sidebar.write("Bezig met analyseren...")
    # Hier kun je de Excel logica plaatsen zoals in het vorige bericht
