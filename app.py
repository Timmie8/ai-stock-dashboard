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

# Initialiseer de watchlist in het geheugen
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "GOOGL"]

# --- 2. ROBUUSTE DATA FUNCTIE ---
def get_finnhub_data(ticker):
    base_url = "https://finnhub.io/api/v1"
    params = {'symbol': ticker, 'token': API_KEY}
    
    try:
        # Quote (Huidige prijs)
        q = requests.get(f"{base_url}/quote", params=params).json()
        
        # Candles (Historische data voor AI)
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=120)).timestamp())
        c_params = {**params, 'resolution': 'D', 'from': start, 'to': end}
        c = requests.get(f"{base_url}/stock/candle", params=c_params).json()
        
        # Metrics (Earnings & Fundamenten)
        f_params = {**params, 'metric': 'all'}
        f = requests.get(f"{base_url}/stock/metric", params=f_params).json()
        
        # Sentiment & Insider data
        s = requests.get(f"{base_url}/news-sentiment", params=params).json()
        insider = requests.get(f"{base_url}/stock/insider-transactions", params=params).json()
        
        return q, c, s, f, insider
    except Exception as e:
        st.error(f"Verbindingsfout met API: {e}")
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
    
    # De grote analyse knop
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")
    
    if st.button("🗑️ Wis Watchlist"):
        st.session_state.watchlist = ["AAPL"]
        st.rerun()

# --- 4. HOOFD DASHBOARD ---
st.title("🛡️ AI Finnhub Professional Terminal")

if analyze_btn:
    with st.spinner(f'Gegevens verzamelen voor {selected_stock}...'):
        q, c, s, f, insider = get_finnhub_data(selected_stock)
        
        # Controleer of we tenminste de basisprijs hebben
        if q and 'c' in q and q['c'] != 0:
            current_price = q['c']
            
            # AI Voorspelling Logica met Fallback
            has_ai = False
            pred_price = current_price
            
            if c and 's' in c and c['s'] == 'ok':
                try:
                    df_ai = pd.DataFrame({'price': c['c']})
                    df_ai['next'] = df_ai['price'].shift(-1)
                    train_df = df_ai.dropna()
                    
                    if not train_df.empty:
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(train_df[['price']], train_df['next'])
                        pred_price = model.predict([[current_price]])[0]
                        has_ai = True
                except:
                    pass

            # Sentiment verwerken
            bullish_pct = 0.5
            if s and isinstance(s, dict) and 'sentiment' in s:
                bullish_pct = s['sentiment'].get('bullishPercent', 0.5) or 0.5
            
            # AI Score berekening (0-100)
            price_move = (pred_price - current_price) / current_price if has_ai else 0
            ai_score = (bullish_pct * 100 * 0.4) + (50 + (price_move * 1000)) * 0.6
            ai_score = min(max(ai_score, 0), 100)

            # --- UI: METRIEKEN ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Live Prijs", f"${current_price:.2f}", f"{q.get('d', 0):.2f}")
            
            # Earnings uit financials halen
            next_earn = f.get('metric', {}).get('earningsReleaseDate', 'Niet bekend')
            m2.metric("Earnings Verwacht", next_earn)
            m3.metric("AI Score", f"{ai_score:.1f}/100")

            # --- UI: GRAFIEK & RATING ---
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                if c and 'c' in c:
                    fig = go.Figure(data=[go.Candlestick(
                        x=list(range(len(c['c']))),
                        open=c['o'], high=c['h'], low=c['l'], close=c['c'],
                        name="Candlesticks"
                    )])
                    fig.update_layout(template="plotly_dark", title=f"Technische Analyse: {selected_stock}", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Geen historische kaarsdata beschikbaar voor grafiek.")

            with col_right:
                st.write("### 🤖 AI Rating")
                if ai_score >= 70: st.success("📈 STERK KOOP")
                elif ai_score <= 30: st.error("⚠️ VERKOOP")
                else: st.info("⚖️ NEUTRAAL")
                
                st.write(f"**Sentiment:** {bullish_pct*100:.1f}% Bullish")
                if has_ai:
                    st.write(f"**AI Target (24u):** ${pred_price:.2f}")
                
                st.markdown("---")
                st.write("### 👥 Insider Activity")
                if insider and 'data' in insider and len(insider['data']) > 0:
                    ins_df = pd.DataFrame(insider['data']).head(5)
                    st.dataframe(ins_df[['name', 'share', 'change']], hide_index=True)
                else:
                    st.caption("Geen recente insider transacties gevonden.")

            # --- EXCEL EXPORT ---
            st.markdown("---")
            report_df = pd.DataFrame([{
                "Ticker": selected_stock, "Prijs": current_price, 
                "AI Score": ai_score, "Sentiment": bullish_pct, 
                "Datum": datetime.now().strftime("%Y-%m-%d %H:%M")
            }])
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                report_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Download Analyse Rapport (Excel)",
                data=output.getvalue(),
                file_name=f"AI_Report_{selected_stock}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error(f"Finnhub kon geen koersdata vinden voor '{selected_stock}'. Controleer de ticker of probeer later opnieuw.")
else:
    st.info("👈 Gebruik de sidebar om een aandeel te kiezen en klik op 'START AI ANALYSE'.")





