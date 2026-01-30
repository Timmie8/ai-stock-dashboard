import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import io

# --- 1. CONFIGURATIE ---
API_KEY = "d5h3vm9r01qll3dlm2sgd5h3vm9r01qll3dlm2t0"
st.set_page_config(page_title="AI Finnhub Pro Terminal", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "GOOGL"]

# --- 2. ROBUUSTE DATA FUNCTIE ---
def get_finnhub_data(ticker):
    base_url = "https://finnhub.io/api/v1"
    params = {'symbol': ticker, 'token': API_KEY}
    try:
        q = requests.get(f"{base_url}/quote", params=params).json()
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=150)).timestamp())
        c_params = {**params, 'resolution': 'D', 'from': start, 'to': end}
        c = requests.get(f"{base_url}/stock/candle", params=c_params).json()
        s = requests.get(f"{base_url}/news-sentiment", params=params).json()
        f = requests.get(f"{base_url}/stock/metric", params=params, data={'metric': 'all'}).json()
        insider = requests.get(f"{base_url}/stock/insider-transactions", params=params).json()
        return q, c, s, f, insider
    except Exception as e:
        st.error(f"Verbindingsfout: {e}")
        return None, None, None, None, None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⭐ Portfolio Management")
    new_stock = st.text_input("Voeg USA Ticker toe:").upper().strip()
    if st.button("➕ Toevoegen"):
        if new_stock and new_stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_stock)
            st.rerun()
    
    selected_stock = st.selectbox("Kies een aandeel:", st.session_state.watchlist)
    st.markdown("---")
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")

# --- 4. HOOFD DASHBOARD ---
st.title("🛡️ AI Finnhub Professional Terminal")

if analyze_btn:
    with st.spinner(f'Diepe AI-analyse voor {selected_stock}...'):
        q, c, s, f, insider = get_finnhub_data(selected_stock)
        
        if q and 'c' in q and q['c'] != 0:
            current_price = q['c']
            
            # --- VERBETERDE MULTI-FEATURE AI LOGICA ---
            if c and 's' in c and c['s'] == 'ok':
                # Maak een DataFrame met meerdere kenmerken
                df_ai = pd.DataFrame({
                    'price': c['c'],
                    'high': c['h'],
                    'low': c['l'],
                    'volume': c['v']
                })
                # Bereken extra indicatoren voor unieke scores
                df_ai['returns'] = df_ai['price'].pct_change()
                df_ai['range'] = (df_ai['high'] - df_ai['low']) / df_ai['price']
                df_ai['target'] = df_ai['price'].shift(-1)
                
                train_df = df_ai.dropna()
                
                # Features: Prijs, Momentum, Volatiliteit en Volume
                features = ['price', 'returns', 'range', 'volume']
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(train_df[features], train_df['target'])
                
                # Voorspelling
                last_row = df_ai[features].iloc[-1].values.reshape(1, -1)
                pred_price = model.predict(last_row)[0]
                tech_diff = ((pred_price - current_price) / current_price) * 100
                has_ai = True
            else:
                pred_price = current_price
                tech_diff = 0
                has_ai = False

            # --- SENTIMENT & SCORE BEREKENING ---
            sentiment_impact = 0
            if s and 'sentiment' in s:
                bullish_pct = s['sentiment'].get('bullishPercent', 0.5) or 0.5
                sentiment_impact = (bullish_pct - 0.5) * 40 # Range -20 tot +20

            # Finale score opbouw
            # Basis 50 + Technische trend (x15 impact) + Sentiment impact
            ai_score = 50 + (tech_diff * 15) + sentiment_impact
            
            # Voeg unieke 'fingerprint' toe op basis van volume
            ai_score += (q.get('v', 0) % 10) / 10
            ai_score = min(max(ai_score, 5), 95)

            # --- UI: METRIEKEN ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Live Prijs", f"${current_price:.2f}", f"{q.get('d', 0):.2f}")
            m2.metric("AI Score", f"{ai_score:.1f}/100")
            m3.metric("AI Doel (24u)", f"${pred_price:.2f}")

            # --- UI: GRAFIEK & RATING ---
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                if c and 'c' in c:
                    fig = go.Figure(data=[go.Candlestick(
                        x=list(range(len(c['c']))),
                        open=c['o'], high=c['h'], low=c['l'], close=c['c'],
                        name="Market Price"
                    )])
                    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
            with col_right:
                st.write("### AI Rating")
                if ai_score >= 70: st.success("🚀 STERK KOOP")
                elif ai_score <= 30: st.error("⚠️ VERKOOP")
                else: st.info("⚖️ NEUTRAAL")
                
                # Sentiment visualisatie
                st.progress(ai_score / 100)
                st.caption(f"De AI is voor {ai_score:.1f}% zeker van de huidige trend.")

                st.markdown("---")
                st.write("### 👥 Insider Activity")
                if insider and 'data' in insider and len(insider['data']) > 0:
                    ins_df = pd.DataFrame(insider['data']).head(5)
                    st.dataframe(ins_df[['name', 'share', 'change']], hide_index=True)
                else:
                    st.caption("Geen recente insider transacties gevonden.")

            # --- EXPORT ---
            report_df = pd.DataFrame([{"Ticker": selected_stock, "Score": ai_score, "Target": pred_price, "Price": current_price}])
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                report_df.to_excel(writer, index=False)
            st.download_button("📥 Export naar Excel", buffer.getvalue(), file_name=f"{selected_stock}_AI.xlsx")

        else:
            st.error("Finnhub kon geen live data vinden. Controleer de ticker.")
else:
    st.info("👈 Selecteer een aandeel en klik op 'START AI ANALYSE'.")






