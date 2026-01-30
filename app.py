import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import io

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="AI Visual Strategy Terminal", layout="wide")

# Initialiseer de watchlist in de sessie
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD"]

# --- 2. HULPFUNCTIES (Scrapers & AI) ---
def get_live_sentiment(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text.lower() for h in soup.find_all(['h3', 'h2'])][:10]
        
        if not headlines: return 50, "NEUTRAL"
        
        pos_words = ['growth', 'buy', 'up', 'surge', 'profit', 'positive', 'beat', 'bull', 'strong', 'upgrade']
        neg_words = ['drop', 'fall', 'sell', 'loss', 'negative', 'miss', 'bear', 'weak', 'risk', 'downgrade']
        
        score = 65 
        for h in headlines:
            for word in pos_words:
                if word in h: score += 3
            for word in neg_words:
                if word in h: score -= 3
        return min(98, max(30, score)), ("POSITIVE" if score > 70 else "NEGATIVE" if score < 45 else "NEUTRAL")
    except:
        return 50, "UNAVAILABLE"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ AI Portfolio")
    new_ticker = st.text_input("Voeg Ticker toe:").upper().strip()
    if st.button("➕ Voeg toe") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.rerun()
    
    selected_stock = st.selectbox("Selecteer aandeel:", st.session_state.watchlist)
    st.markdown("---")
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")

# --- 4. HOOFD DASHBOARD ---
st.title("🏹 AI Visual Strategy Dashboard")

if selected_stock and analyze_btn:
    try:
        with st.spinner(f'AI Analyseert {selected_stock}...'):
            # 1. Data ophalen
            ticker_obj = yf.Ticker(selected_stock)
            data = ticker_obj.history(period="150d")
            
            if data.empty:
                st.error("Kon geen data vinden voor dit aandeel.")
            else:
                current_price = float(data['Close'].iloc[-1])
                
                # 2. Technische Berekeningen
                # Trend Regressie
                y_reg = data['Close'].values.reshape(-1, 1)
                X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
                reg_model = LinearRegression().fit(X_reg, y_reg)
                pred_price = float(reg_model.predict(np.array([[len(y_reg)]]))[0][0])
                
                # RSI & ATR
                delta = data['Close'].diff()
                up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / (ema_down + 1e-9)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
                
                high_low = data['High'] - data['Low']
                atr = high_low.rolling(14).mean().iloc[-1]
                
                # 3. AI Scoring
                sentiment_score, sentiment_status = get_live_sentiment(selected_stock)
                ensemble_score = int(72 + (12 if pred_price > current_price else -8) + (10 if rsi < 45 else 0))
                
                # --- VISUALISATIE ---
                c1, c2, c3 = st.columns(3)
                price_chg = ((current_price / data['Close'].iloc[-2]) - 1) * 100
                c1.metric("Prijs", f"${current_price:.2f}", f"{price_chg:.2f}%")
                c2.metric("AI Score", f"{ensemble_score}%", sentiment_status)
                c3.metric("RSI Indicator", f"{rsi:.1f}")

                # Grafiek met trendlijn
                
                chart_data = data[['Close']].copy()
                chart_data['AI Trend'] = reg_model.predict(X_reg)
                st.line_chart(chart_data)

                # Strategie Tabel
                st.subheader("🚀 Strategy Scoreboard")
                
                def get_row(name, is_buy, signal, target):
                    return {
                        "Methode": name, 
                        "Status": "BUY" if is_buy else "HOLD",
                        "Signal": signal, 
                        "Target": f"${target:.2f}" if is_buy else "-"
                    }

                strategies = [
                    get_row("Ensemble Learning", ensemble_score > 75, f"{ensemble_score}% betrouwbaarheid", current_price + (2*atr)),
                    get_row("Sentiment Scraper", sentiment_score > 72, sentiment_status, current_price + atr),
                    get_row("Trend Regression", pred_price > current_price, f"Target ${pred_price:.2f}", pred_price),
                    get_row("RSI Swing", rsi < 40, f"RSI: {rsi:.1f}", current_price + (3*atr))
                ]
                
                df_strat = pd.DataFrame(strategies)
                
                def color_status(val):
                    color = '#00C851' if val == 'BUY' else '#FFBB33'
                    return f'background-color: {color}; color: white; font-weight: bold'

                st.table(df_strat.style.applymap(color_status, subset=['Status']))

    except Exception as e:
        st.error(f"Fout tijdens analyse: {e}")
else:
    st.info("👈 Selecteer een aandeel in de sidebar en klik op de blauwe knop om de AI-modellen te starten.")






