import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
# ... (overige imports blijven gelijk)

# --- FUNCTIE: BATCH EXPORT ---
def generate_excel_report(watchlist):
    report_data = []
    
    progress_bar = st.sidebar.progress(0)
    for i, ticker in enumerate(watchlist):
        try:
            # Data ophalen
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df.empty: continue
            
            # AI Score berekenen (versnelde versie voor export)
            current_price = df['Close'].iloc[-1]
            news = stock.news
            sent_score = analyze_sentiment(news)
            
            # Simpele trend indicator voor het rapport
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
            
            # De score logica
            final_score = min(max(50 + (price_change * 2) + (sent_score * 20), 0), 100)
            
            report_data.append({
                "Ticker": ticker,
                "Huidige Prijs": round(current_price, 2),
                "AI Score": round(final_score, 1),
                "Sentiment": round(sent_score, 2),
                "20d Trend (%)": round(price_change, 2),
                "Datum": datetime.now().strftime("%d-%m-%Y")
            })
        except Exception as e:
            st.error(f"Fout bij {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(watchlist))
    
    return pd.DataFrame(report_data)

# --- SIDEBAR UPGRADE ---
with st.sidebar:
    st.markdown("---")
    st.header("📂 Rapportage")
    if st.button("Genereer Watchlist Rapport"):
        with st.spinner("Alle aandelen worden geanalyseerd..."):
            report_df = generate_excel_report(st.session_state.watchlist)
            
            # Excel buffer maken
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                report_df.to_excel(writer, index=False, sheet_name='AI_Scores')
            
            st.download_button(
                label="📥 Download Excel Rapport",
                data=buffer.getvalue(),
                file_name=f"Stock_AI_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
