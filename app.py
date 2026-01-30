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

# --- 1. CONFIGURATIE & INITIALISATIE ---
st.set_page_config(page_title="USA AI Stock Analyzer", layout="wide")

# Zorg dat de watchlist altijd bestaat
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD"]

# Download NLTK data voor sentiment analyse
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

# --- 2. DATA FUNCTIES ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 2 jaar data voor betere AI training
        df = stock.history(period="2y")
        if df.empty:
            return None, None, None
        
        news = stock.news
        
        # Earnings datum ophalen
        calendar = stock.calendar
        next_earn = "Onbekend"
        if calendar is not None and not calendar.empty:
            try:
                # Voor USA aandelen staat de datum vaak in de eerste kolom
                next_earn = calendar.iloc[0, 0].strftime('%d-%m-%Y')
            except:
                next_earn = "Binnenkort"
                
        return df, news, next_earn
    except:
        return None, None, None

def run_ai_prediction(df):
    # Ensemble Model (Random Forest)
    df_clean = df[['Close']].copy()
    df_clean['Target'] = df_clean['Close'].shift(-1)
    train_data = df_clean.dropna()
    
    X = train_data[['Close']].values
    y = train_data['Target'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_price = df['Close'].iloc[-1]
    prediction = model.predict([[last_price]])[0]
    
    return prediction, last_price

def get_sentiment_score(news):
    if not news:
        return 0
    scores = []
    for article in news[:10]:
        s = sia.polarity_scores(article['title'])['compound']
        scores.append(s)
    return np.mean(scores)

# --- 3. SIDEBAR (Watchlist) ---
with st.sidebar:
    st.title("📂 Portfolio")
    new_stock = st.text_input("Voeg USA Ticker toe (bijv. GOOGL):").upper()
    if st.button("Toevoegen"):
        if new_stock and new_stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_stock)
            st.rerun()
    
    selected_stock = st.selectbox("Selecteer aandeel:", st.session_state.watchlist)
    
    st.markdown("---")
    if st.button("Rapport exporteren (Excel)"):
        st.write("Functie wordt voorbereid...")

# --- 4. HOOFD DASHBOARD ---
st.title(f"📊 AI Dashboard: {selected_stock if selected_stock else 'Selecteer aandeel'}")

if selected_stock:
    df, news, earnings = get_stock_data(selected_stock)
    
    if df is not None:
        pred_price, current_price = run_ai_prediction(df)
        sentiment = get_sentiment_score(news)
        
        # AI Score Berekening
        # Combineert prijsactie (70%) en sentiment (30%)
        price_move = (pred_price - current_price) / current_price
        ai_score = 50 + (price_move * 500) + (sentiment * 20)
        ai_score = min(max(ai_score, 0), 100)
        
        # Statistieken rij
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Huidige Koers", f"${current_price:.2f}")
        col2.metric("AI Doel (24u)", f"${pred_price:.2f}")
        col3.metric("Earnings Datum", earnings)
        col4.metric("AI Score", f"{ai_score:.1f}/100")
        
        # Signaal Sectie
        if ai_score >= 70:
            st.success("🎯 **BUY SIGNAL**: De AI verwacht een opwaartse trend ondersteund door sentiment.")
        elif ai_score <= 30:
            st.error("⚠️ **SELL SIGNAL**: De modellen wijzen op een daling of negatief sentiment.")
        else:
            st.info("⚖️ **HOLD**: Geen sterke afwijking gedetecteerd.")
            
        # Grafiek
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'][-100:], name="Historie", line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=[df.index[-1] + timedelta(days=1)], y=[pred_price], 
                                 mode='markers', marker=dict(size=12, color='gold'), name="AI Voorspelling"))
        fig.update_layout(title="Koersverloop (Laatste 100 dagen)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Management
        st.subheader("🛡️ Trade Planner")
        c_risk1, c_risk2 = st.columns(2)
        with c_risk1:
            vol = df['Close'].pct_change().std()
            sl = current_price * (1 - (vol * 2))
            tp = current_price * (1 + (vol * 4))
            st.write(f"**Aanbevolen Stop-Loss:** ${sl:.2f}")
            st.write(f"**Aanbevolen Take-Profit:** ${tp:.2f}")
            
        # Nieuws
        st.subheader("📰 Relevant Nieuws")
        for a in news[:3]:
            st.markdown(f"* [{a['title']}]({a['link']}) ({a['publisher']})")
            
    else:
        st.error(f"Kon geen data vinden voor {selected_stock}. Controleer of de ticker correct is op Yahoo Finance.")

