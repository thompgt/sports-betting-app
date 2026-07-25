import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

from app.core.config import get_settings

# Page Config
st.set_page_config(page_title="LineEdge - Sports Betting Analytics", layout="wide")

# Database Connection (same DB the detection service writes to)
settings = get_settings()
engine = create_engine(settings.db_url)

def load_data():
    try:
        df = pd.read_sql("SELECT * FROM detected_edges", engine)
        df['timestamp_detected'] = pd.to_datetime(df['timestamp_detected'])
        return df
    except Exception:
        return pd.DataFrame()

st.title("🎯 LineEdge Dashboard")
st.markdown("### Multi-Bookmaker Arbitrage & EV Detection Engine")

df = load_data()

if df.empty:
    st.warning("No data found in the database. Run the simulation script first!")
else:
    tab1, tab2, tab3 = st.tabs(["🔥 Live Market Edges", "📊 Historical Analytics", "⚖️ CLV Audit"])

    with tab1:
        st.subheader("Current Positive Expected Value (+EV) Opportunities")
        
        # Filter for active edges
        active_edges = df[df['is_active'] == 1].copy()
        active_edges = active_edges.sort_values(by='calculated_ev', ascending=False)
        
        if not active_edges.empty:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Edges", len(active_edges))
            col2.metric("Max EV%", f"{active_edges['calculated_ev'].max():.2%}")
            col3.metric("Avg EV%", f"{active_edges['calculated_ev'].mean():.2%}")

            # Styled Table
            st.dataframe(
                active_edges[[
                    'timestamp_detected', 'sport', 'bookmaker_name', 'outcome_name',
                    'odds_offered', 'fair_odds', 'calculated_ev'
                ]].style.format({
                    'calculated_ev': '{:.2%}',
                    'odds_offered': '{:.2f}',
                    'fair_odds': '{:.2f}'
                }).background_gradient(subset=['calculated_ev'], cmap='Greens'),
                use_container_width=True
            )
        else:
            st.info("No active edges currently detected.")

    with tab2:
        st.subheader("Edge Detection Performance")
        
        # Edges over time
        df_time = df.set_index('timestamp_detected').resample('h').size().reset_index(name='count')
        fig_time = px.line(df_time, x='timestamp_detected', y='count', title="Edges Detected Over Time")
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Edges by sport
        fig_sport = px.bar(
            df.groupby('sport').size().reset_index(name='count'),
            x='sport', y='count', color='sport',
            title="Total Edges Found by Sport"
        )
        st.plotly_chart(fig_sport, use_container_width=True)

    with tab3:
        st.subheader("Closing Line Value (CLV) Analysis")
        st.markdown("This section tracks how our predicted 'Fair Odds' compare to the market's final closing lines, recorded by the EdgeAuditor once each game starts.")

        clv_df = df[df['closing_line'].notna()].copy()

        if clv_df.empty:
            st.info("No closed-out markets yet. CLV data appears once games in the database have started and the detection service has run a poll cycle.")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Markets Closed Out", len(clv_df))
            col2.metric("Avg CLV%", f"{clv_df['clv_pct'].mean():.2%}")

            fig_clv = px.scatter(
                clv_df, x='fair_odds', y='closing_line',
                trendline="ols",
                title="Fair Price vs Market Closing Price"
            )
            st.plotly_chart(fig_clv, use_container_width=True)
            st.write("A strong correlation here proves that our devigging engine identifies the 'True' market price before the books adjust.")
