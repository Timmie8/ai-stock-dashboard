import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. CONFIGURATIE ---
API_KEY = "d5h3vm9r01qll3dlm2sgd5h3vm9r01qll3dlm2t0"
st.set_page_config(page_title="AI Finnhub Pro Terminal", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT"]

# --- 2. FINNHUB API FUNCTIES ---
def get_finnhub_data(ticker):
    base_url = "https://finnhub.io/api/v1"
    q = requests.get(f"{base_url}/quote?symbol={ticker}&token={API_KEY}").json()
    
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=100)).timestamp())
    c = requests.get(f"{base_url}/stock/candle?symbol={ticker}&resolution=D&from={start}&to={end}&token={API_KEY}").json()
    
    s = requests.get(f"{base_url}/news-sentiment?symbol={ticker}&token={API_KEY}").json()
    f = requests.get(f"{base_url}/stock/metric?symbol={ticker}&metric=all&token={API_KEY}").json()
    
    # NIEUW: Insider Transactions
    insider = requests.get(f"{base_url}/stock/insider-transactions?symbol={ticker}&token={API_KEY}").json()
    
    return q, c, s, f, insider

# --- 3. DASHBOARD UI ---
st.title("🛡️ AI Finnhub Professional Terminal")

with st.sidebar:
    st.header("⭐ Portfolio Management")
    new_stock = st.text_input("Voeg Ticker toe:").upper()
    if st.button("➕ Toevoegen"):
        if new_stock and new_stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_stock)
            st.rerun()
    
    selected_stock = st.selectbox("Selecteer aandeel uit lijst:", st.session_state.watchlist)
    
    st.markdown("---")
    # DE ANALYSE KNOP
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")

# Logica wanneer er op de knop wordt gedrukt
if analyze_btn or 'current_analysis' in st.session_state:
    # We slaan de huidige analyse op in de session_state zodat hij niet verdwijnt bij interactie
    st.session_state.current_analysis = selected_stock
    
    with st.spinner(f'AI analyseert {selected_stock}...'):
        q, c, s, f, insider = get_finnhub_data(selected_stock)
        
        if q and 'c' in q and q['c'] != 0:
            current_price = q['c']
            
            # Sentiment veiligstellen
            bullish_pct = s.get('sentiment', {}).get('bullishPercent', 0.5) if s else 0.5
            if bullish_pct is None: bullish_pct = 0.5

            # AI Model
            df = pd.DataFrame({'price': c['c']})
            df['next'] = df['price'].shift(-1)
            model = RandomForestRegressor(n_estimators=50).fit(df[['price']].iloc[:-1], df['next'].iloc[:-1])
            pred_price = model.predict([[current_price]])[0]

            # Score
            ai_score = (bullish_pct * 100 * 0.4) + (50 + (pred_price - current_price) / current_price * 1000) * 0.6
            ai_score = min(max(ai_score, 0), 100)

            # --- DISPLAY ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Live Prijs", f"${current_price:.2f}", f"{q['d']:.2f}")
            col2.metric("Earnings Verwacht", f.get('metric', {}).get('earningsReleaseDate', 'N/A'))
            col3.metric("AI Score", f"{ai_score:.1f}/100")

            # Grafiek & Rating
            l_col, r_col = st.columns([2, 1])
            with l_col:
                fig = go.Figure(data=[go.Candlestick(x=list(range(len(c['c']))),
                                open=c['o'], high=c['h'], low=c['l'], close=c['c'])])
                fig.update_layout(template="plotly_dark", title=f"Technische Analyse: {selected_stock}")
                st.plotly_chart(fig, use_container_width=True)
            
            with r_col:
                st.write("### AI Advies")
                if ai_score >= 70: st.success("📈 STERK KOOP")
                elif ai_score <= 30: st.error("⚠️ VERKOOP")
                else: st.info("⚖️ NEUTRAAL")
                
                # Insider Transactions tabel
                st.write("### 👥 Insider Activity")
                if insider and 'data' in insider:
                    ins_df = pd.DataFrame(insider['data']).head(5)
                    if not ins_df.empty:
                        st.table(ins_df[['name', 'share', 'change']])
                    else:
                        st.write("Geen recente transacties.")

        else:
            st.error("Data ophalen mislukt. Controleer de ticker.")
else:
    st.info("👈 Selecteer een aandeel en klik op de knop 'START AI ANALYSE' in de sidebar.")




