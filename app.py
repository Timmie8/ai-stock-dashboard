import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- 1. CONFIGURATIE ---
st.set_page_config(page_title="AI Pro Strategy Terminal", layout="wide")

# Custom CSS voor de gekleurde boxen
st.markdown("""
    <style>
    .metric-container {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #333;
        min-height: 110px;
    }
    .metric-green {
        border-left: 5px solid #00C851 !important;
        background-color: #06402B !important;
    }
    </style>
    """, unsafe_allow_html=True)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD"]

# --- 2. EARNINGS FUNCTIE ---
def get_earnings_info(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                return cal['Earnings Date'][0].strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                return cal.iloc[0, 0].strftime('%Y-%m-%d')
        info = ticker_obj.info
        if 'nextEarningsDate' in info:
            return datetime.fromtimestamp(info['nextEarningsDate']).strftime('%Y-%m-%d')
        return "Binnenkort"
    except:
        return "Check Yahoo"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ AI Portfolio")
    new_ticker = st.text_input("Voeg Ticker toe:").upper().strip()
    if st.button("➕ Voeg toe") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.rerun()
    selected_stock = st.selectbox("Selecteer aandeel:", st.session_state.watchlist)
    analyze_btn = st.button("🚀 START AI ANALYSE", use_container_width=True, type="primary")

# --- 4. HOOFD DASHBOARD ---
if selected_stock and analyze_btn:
    try:
        with st.spinner(f'Modellen worden geladen voor {selected_stock}...'):
            ticker_obj = yf.Ticker(selected_stock)
            data = ticker_obj.history(period="200d")
            
            if data.empty:
                st.error("Geen data gevonden.")
            else:
                current_price = float(data['Close'].iloc[-1])
                earnings_date = get_earnings_info(ticker_obj)
                
                # --- ANALYSE BEREKENINGEN ---
                y_reg = data['Close'].values.reshape(-1, 1)
                X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
                reg_model = LinearRegression().fit(X_reg, y_reg)
                pred_price = float(reg_model.predict(np.array([[len(y_reg)]]))[0][0])
                
                last_5_days = data['Close'].iloc[-5:].pct_change().sum()
                momentum_score = int(68 + (last_5_days * 160))
                
                delta = data['Close'].diff()
                up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / (ema_down + 1e-9)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
                
                ensemble_score = int(70 + (10 if pred_price > current_price else -10) + (12 if rsi < 45 else 0))

                recent_high = float(data['High'].iloc[-22:-1].max())
                sma50 = float(data['Close'].rolling(window=50).mean().iloc[-1])
                atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]

                # --- TECH SIGNAL LOGICA VOOR BOVENBALK ---
                tech_label = "HOLD"
                tech_buy = False
                
                if current_price >= recent_high:
                    tech_label = "BUY (Breakout)"
                    tech_buy = True
                elif rsi < 35:
                    tech_label = "BUY (Swing)"
                    tech_buy = True
                elif pred_price > current_price:
                    tech_label = "BUY (Trend)"
                    tech_buy = True
                elif current_price < (sma50 * 0.93):
                    tech_label = "BUY (Reversal)"
                    tech_buy = True

                # --- BOVENSTE RIJ: METRIC CARDS ---
                st.subheader(f"Dashboard: {selected_stock}")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                price_chg = ((current_price / data['Close'].iloc[-2]) - 1) * 100
                col1.metric("Koers", f"${current_price:.2f}", f"{price_chg:.2f}%")
                col2.metric("Earnings", earnings_date)
                
                # AI Ensemble Card
                ens_class = "metric-green" if ensemble_score >= 75 else ""
                col3.markdown(f"""<div class="metric-container {ens_class}">
                    <p style='margin:0;font-size:14px;color:#aaa;'>AI Ensemble</p>
                    <h2 style='margin:0;color:white;'>{ensemble_score}%</h2>
                    </div>""", unsafe_allow_html=True)
                
                # Momentum AI Card
                mom_class = "metric-green" if momentum_score >= 75 else ""
                col4.markdown(f"""<div class="metric-container {mom_class}">
                    <p style='margin:0;font-size:14px;color:#aaa;'>Momentum AI</p>
                    <h2 style='margin:0;color:white;'>{momentum_score}%</h2>
                    </div>""", unsafe_allow_html=True)
                
                # Tech Signal Card (Nu met dynamische naam)
                tech_class = "metric-green" if tech_buy else ""
                col5.markdown(f"""<div class="metric-container {tech_class}">
                    <p style='margin:0;font-size:14px;color:#aaa;'>Tech Signal</p>
                    <h2 style='margin:0;color:white;font-size:20px;'>{tech_label}</h2>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")
                
                # Grafiek
                
                st.line_chart(data[['Close']])

                # --- STRATEGIE TABEL ---
                st.subheader("🚀 Comprehensive Strategy Scoreboard")
                
                strategies = [
                    {"Categorie": "AI", "Methode": "Ensemble Learning", "Score": ensemble_score, "Status": "BUY" if ensemble_score >= 75 else "HOLD", "Target": f"${current_price + (3*atr):.2f}"},
                    {"Categorie": "AI", "Methode": "Momentum AI", "Score": momentum_score, "Status": "BUY" if momentum_score >= 75 else "HOLD", "Target": f"${current_price + (4*atr):.2f}"},
                    {"Categorie": "Tech", "Methode": "Trend Regressie", "Score": 82 if pred_price > current_price else 45, "Status": "BUY" if pred_price > current_price else "HOLD", "Target": f"${pred_price:.2f}"},
                    {"Categorie": "Tech", "Methode": "Swingtrade (RSI)", "Score": 85 if rsi < 35 else 50, "Status": "BUY" if rsi < 35 else "HOLD", "Target": f"${recent_high:.2f}"},
                    {"Categorie": "Tech", "Methode": "Breakout", "Score": 90 if current_price >= recent_high else 40, "Status": "BUY" if current_price >= recent_high else "HOLD", "Target": f"${current_price + (3*atr):.2f}"},
                    {"Categorie": "Tech", "Methode": "Reversal", "Score": 80 if current_price < (sma50 * 0.93) else 45, "Status": "BUY" if current_price < (sma50 * 0.93) else "HOLD", "Target": f"${sma50:.2f}"}
                ]
                
                df_strat = pd.DataFrame(strategies)

                def style_row(row):
                    if row.Score >= 75:
                        return ['background-color: #00C851; color: white; font-weight: bold'] * len(row)
                    return [''] * len(row)

                st.table(df_strat.style.apply(style_row, axis=1))

    except Exception as e:
        st.error(f"Fout bij analyse: {e}")











