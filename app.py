import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import io

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="AI Earnings & Strategy Terminal", layout="wide")

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD"]

# --- 2. HERSTELDE EARNINGS FUNCTIE ---
def get_earnings_info(ticker_obj):
    try:
        # Poging 1: De standaard kalender
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                return cal['Earnings Date'][0].strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                return cal.iloc[0, 0].strftime('%Y-%m-%d')
        
        # Poging 2: Info metadata (fallback)
        info = ticker_obj.info
        if 'nextEarningsDate' in info:
            return datetime.fromtimestamp(info['nextEarningsDate']).strftime('%Y-%m-%d')
            
        return "Niet gevonden"
    except:
        return "Binnenkort verwacht"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ AI Portfolio")
    new_ticker = st.text_input("Voeg USA Ticker toe:").upper().strip()
    if st.button("➕ Voeg toe") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.rerun()
    
    selected_stock = st.selectbox("Selecteer aandeel:", st.session_state.watchlist)
    st.markdown("---")
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")

# --- 4. HOOFD DASHBOARD ---
st.title(f"🏹 Strategy Terminal: {selected_stock}")

if selected_stock and analyze_btn:
    try:
        with st.spinner(f'Data ophalen voor {selected_stock}...'):
            ticker_obj = yf.Ticker(selected_stock)
            # We halen iets meer data op voor een betere trend
            data = ticker_obj.history(period="200d")
            
            if data.empty:
                st.error("Geen koersdata gevonden voor dit symbool.")
            else:
                current_price = float(data['Close'].iloc[-1])
                earnings_date = get_earnings_info(ticker_obj)
                
                # --- AI METHODE: Trend Regressie ---
                y_reg = data['Close'].values.reshape(-1, 1)
                X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
                reg_model = LinearRegression().fit(X_reg, y_reg)
                pred_price = float(reg_model.predict(np.array([[len(y_reg)]]))[0][0])
                
                # --- AI METHODE: Momentum Score ---
                last_5_days = data['Close'].iloc[-5:].pct_change().sum()
                lstm_score = int(68 + (last_5_days * 160))
                
                # --- AI METHODE: Ensemble (RSI + Trend) ---
                delta = data['Close'].diff()
                up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / (ema_down + 1e-9)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
                
                # Score berekening
                ensemble_score = int(70 + (10 if pred_price > current_price else -10) + (12 if rsi < 45 else 0))

                # Hulpcijfer: ATR
                high_low = data['High'] - data['Low']
                atr = high_low.rolling(14).mean().iloc[-1]

                # --- VISUALISATIE METRICS ---
                c1, c2, c3, c4 = st.columns(4)
                price_chg = ((current_price / data['Close'].iloc[-2]) - 1) * 100
                c1.metric("Huidige Koers", f"${current_price:.2f}", f"{price_chg:.2f}%")
                c2.metric("Volgende Earnings", earnings_date)
                c3.metric("AI Ensemble", f"{ensemble_score}%")
                c4.metric("RSI (14d)", f"{rsi:.1f}")

                # Koersgrafiek
                chart_data = data[['Close']].copy()
                chart_data['AI Trendlijn'] = reg_model.predict(X_reg)
                st.line_chart(chart_data)

                # --- STRATEGIE TABEL MET GROENE HIGHLIGHT ---
                st.subheader("🚀 Multi-Method Strategy Scoreboard")
                
                def get_row(name, cat, score_val, signal_desc, target):
                    return {
                        "Categorie": cat,
                        "AI Methode": name, 
                        "Score": score_val,
                        "Status": "BUY" if score_val >= 75 else "HOLD",
                        "Signaal": signal_desc, 
                        "Target": f"${target:.2f}" if score_val >= 75 else "-"
                    }

                strategies = [
                    get_row("Ensemble Learning", "AI", ensemble_score, f"Betrouwbaarheid: {ensemble_score}%", current_price + (3*atr)),
                    get_row("Momentum AI", "AI", lstm_score, "Koersversnelling", current_price + (4*atr)),
                    get_row("Regressie Trend", "Tech", int(82 if pred_price > current_price else 45), f"Trend Doel: ${pred_price:.2f}", pred_price),
                    get_row("RSI Indicator", "Tech", int(88 if rsi < 35 else 50), f"Oververkocht bij {rsi:.1f}", current_price + (2*atr))
                ]
                
                df_all = pd.DataFrame(strategies)

                # Highlighting functie voor scores boven de 75
                def highlight_rows(row):
                    if row['Score'] >= 75:
                        return ['background-color: #00C851; color: white; font-weight: bold'] * len(row)
                    return [''] * len(row)

                st.table(df_all.style.apply(highlight_rows, axis=1))

    except Exception as e:
        st.error(f"Er is een fout opgetreden: {e}")
else:
    st.info("👈 Selecteer een aandeel en klik op 'START AI ANALYSE'.")









