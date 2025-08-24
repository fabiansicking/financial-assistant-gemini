# app.py
# To run this app:
# 1. Save this code as `app.py`
# 2. Save the requirements from the separate block into a `requirements.txt` file
# 3. Open a terminal in the same directory and run:
#    `pip install -r requirements.txt`
#    `streamlit run app.py`

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import random

# --- ML/Forecasting Libraries (Fallbacks are handled if not available)
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter('ignore', ConvergenceWarning)

    STATSMODELS_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    SKLEARN_AVAILABLE = False


# --- GLOBAL SETTINGS
st.set_page_config(
    layout="wide",
    page_title="FinRPG: Personal Financial Assistant",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)
random.seed(42)
np.random.seed(42)

# --- THEME & STYLING
PRIMARY_COLOR = "#8B5CF6"
SECONDARY_COLOR = "#14B8A6"
ACCENT_COLOR = "#3B82F6"
BG_GRADIENT = "linear-gradient(180deg, #0F172A 0%, #1E293B 100%)"
FONT_FAMILY = "Inter, sans-serif"

custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="st-"] {{
        font-family: {FONT_FAMILY};
    }}
    .stApp {{
        background: {BG_GRADIENT};
        color: #E2E8F0;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #E2E8F0;
    }}
    .stMetric {{
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(55, 65, 81, 0.7);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }}
    .stMetric > div:first-child {{
        font-weight: 600;
        color: {SECONDARY_COLOR};
    }}
    .stProgress > div > div {{
        background-color: #374151;
    }}
    .stProgress > div > div > div {{
        background-color: {PRIMARY_COLOR};
    }}
    .stButton > button {{
        background-color: #374151;
        color: #E2E8F0;
        border-radius: 8px;
        border: 1px solid #4B5563;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }}
    .stButton > button:hover {{
        background-color: #4B5563;
        border: 1px solid {PRIMARY_COLOR};
        box-shadow: 0 0 10px {PRIMARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size:1.1rem;
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
    }}
    .st-emotion-cache-1629p8f {{
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        padding: 1rem;
    }}
    .st-emotion-cache-1f874cn {{
        background-color: {SECONDARY_COLOR};
    }}
    .st-emotion-cache-1ky2j6w {{
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- SAMPLE DATA (for download)
SAMPLE_CSV_CONTENT = """date,category,amount,running_balance
2024-01-01,income,1500.00,1500.00
2024-01-02,food,-35.50,1464.50
2024-01-03,bills,-120.00,1344.50
2024-01-04,transport,-22.00,1322.50
2024-01-05,food,-48.75,1273.75
2024-01-08,entertainment,-55.00,1218.75
2024-01-10,income,180.00,1398.75
2024-01-12,bills,-80.00,1318.75
2024-01-15,food,-62.40,1256.35
2024-01-18,transport,-15.50,1240.85
2024-01-20,savings,-200.00,1040.85
2024-01-22,food,-45.00,995.85
2024-01-25,entertainment,-30.00,965.85
2024-01-30,bills,-250.00,715.85
2024-02-01,income,2100.00,2815.85
2024-02-03,food,-40.50,2775.35
2024-02-05,savings,-250.00,2525.35
2024-02-07,transport,-25.00,2500.35
2024-02-10,entertainment,-75.00,2425.35
2024-02-15,food,-55.00,2370.35
2024-02-16,entertainment,-40.00,2330.35
2024-02-20,income,150.00,2480.35
2024-02-25,bills,-100.00,2380.35
2024-02-28,food,-60.00,2320.35
2024-03-01,income,2100.00,4420.35
"""


# --- UTILITY FUNCTIONS
@st.cache_data
def load_and_validate_data(uploaded_file):
    """
    Loads and validates a CSV file, ensuring required columns and data types.
    Returns a DataFrame if valid, otherwise returns None and sets a Streamlit error message.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # 1. Validate columns
        required_cols = ['date', 'category', 'amount', 'running_balance']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Error: The CSV is missing required columns. Please ensure it has: {', '.join(required_cols)}")
            return None
            
        # 2. Validate data types and format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['running_balance'] = pd.to_numeric(df['running_balance'], errors='coerce')

        if df['date'].isnull().any():
            st.error("❌ Error: Invalid date format found. Please use YYYY-MM-DD.")
            return None
        if df['amount'].isnull().any() or df['running_balance'].isnull().any():
            st.error("❌ Error: Non-numeric values found in 'amount' or 'running_balance'.")
            return None

        # Sort by date to ensure proper processing
        df.sort_values(by='date', inplace=True)
        return df

    except Exception as e:
        st.error(f"❌ An error occurred during file processing: {e}")
        return None

@st.cache_data
def compute_daily_data(df):
    """Computes daily summaries for charts and analysis."""
    daily_data = df.groupby(df['date'].dt.date).agg(
        total_daily_flow=('amount', 'sum'),
        end_of_day_balance=('running_balance', 'last')
    ).reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    return daily_data

@st.cache_data
def get_moving_averages(series, window_sizes):
    """Calculates moving averages for a given series."""
    m_averages = {}
    for window in window_sizes:
        m_averages[f'{window}-day MA'] = series.rolling(window=window).mean()
    return pd.DataFrame(m_averages)

@st.cache_data
def forecast_balance(df, method_toggles):
    """
    Produces a 3-month forecast using an ensemble of models.
    Returns a dictionary of forecast results.
    """
    if not STATSMODELS_AVAILABLE or len(df) < 60:
        st.warning("Forecasting requires `statsmodels` and at least 60 days of data. Skipping.")
        return {}

    projections = {}
    ts_data = df.set_index('date')['running_balance']
    
    # Linear Regression Projection
    if method_toggles.get('Linear Regression'):
        model_name = "Linear Regression"
        try:
            model = LinearRegression()
            X = np.arange(len(ts_data)).reshape(-1, 1)
            model.fit(X, ts_data.values)
            future_X = np.arange(len(ts_data), len(ts_data) + 90).reshape(-1, 1)
            forecast = model.predict(future_X)
            projections[model_name] = {'mean': forecast, 'std': np.std(forecast)}
        except Exception:
            pass

    # Holt-Winters Projection
    if method_toggles.get('Holt-Winters'):
        model_name = "Holt-Winters"
        try:
            fit_holt = ExponentialSmoothing(ts_data, trend='add', seasonal=None).fit()
            forecast = fit_holt.forecast(steps=90)
            projections[model_name] = {'mean': forecast.values, 'std': np.std(forecast.values)}
        except Exception:
            pass

    # Monte Carlo Projection
    if method_toggles.get('Monte Carlo'):
        model_name = "Monte Carlo"
        try:
            daily_returns = df['amount'] / df['running_balance'].shift(1)
            daily_returns.dropna(inplace=True)
            mu, sigma = daily_returns.mean(), daily_returns.std()
            
            simulations = 250
            forecasts = np.zeros((simulations, 90))
            for i in range(simulations):
                prices = [ts_data.iloc[-1]]
                for _ in range(90):
                    daily_change = random.gauss(mu, sigma) if sigma > 0 else mu
                    prices.append(prices[-1] * (1 + daily_change))
                forecasts[i, :] = prices[1:]
            projections[model_name] = {'mean': np.mean(forecasts, axis=0), 'std': np.std(forecasts, axis=0)}
        except Exception:
            pass

    # Ensemble Projection
    if projections:
        all_forecasts = [p['mean'] for p in projections.values() if 'mean' in p]
        if all_forecasts:
            ensemble_mean = np.mean(all_forecasts, axis=0)
            ensemble_std = np.std(all_forecasts, axis=0)
            projections['Ensemble'] = {'mean': ensemble_mean, 'std': ensemble_std}

    return projections

@st.cache_data
def get_daily_challenges():
    """Generates a randomized daily challenge."""
    challenges = [
        "Go one day without buying food or coffee out.",
        "Transfer an extra €10 to your savings account.",
        "Review your last week's spending and find one unnecessary purchase.",
        "Track every single expense, no matter how small, for one day.",
        "Cook a meal at home instead of getting takeout."
    ]
    return random.choice(challenges)

@st.cache_data
def create_advice_trees(df):
    """
    Trains and returns a dictionary of DecisionTree models for generating tips.
    Uses heuristics to create labels for training based on user's financial habits.
    """
    if not SKLEARN_AVAILABLE or len(df) < 50:
        return {}

    advice_trees = {}
    
    daily_df = df.set_index('date').resample('D').agg({
        'amount': 'sum', 'running_balance': 'last',
    }).ffill()
    monthly_df = df.set_index('date').resample('M').agg({
        'income': ('amount', lambda x: x[x > 0].sum()),
        'bills': ('amount', lambda x: abs(x[x < 0][df['category'] == 'bills'].sum())),
        'savings': ('amount', lambda x: abs(x[x < 0][df['category'] == 'savings'].sum())),
        'food': ('amount', lambda x: abs(x[x < 0][df['category'] == 'food'].sum())),
        'entertainment': ('amount', lambda x: abs(x[x < 0][df['category'] == 'entertainment'].sum()))
    }).ffill().dropna()

    if monthly_df.empty:
        return {}

    # Heuristic 1: High food spending
    monthly_df['high_food'] = (monthly_df['food'] / monthly_df['income'] > 0.15).astype(int)
    if 'high_food' in monthly_df.columns and monthly_df['high_food'].sum() > 0:
        try:
            X = monthly_df[['food']].values
            y = monthly_df['high_food'].values
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X, y)
            advice_trees['high_food_spend'] = {
                'model': clf, 'features': ['food_spending'], 'condition': clf.predict([[monthly_df['food'].iloc[-1]]])[0] == 1,
                'tip': "Your food expenses are quite high. Try meal-prepping or planning your meals to save money.",
            }
        except Exception: pass

    # Heuristic 2: Low savings rate
    monthly_df['low_savings'] = (monthly_df['savings'] / monthly_df['income'] < 0.10).astype(int)
    if 'low_savings' in monthly_df.columns and monthly_df['low_savings'].sum() > 0:
        try:
            X = monthly_df[['savings']].values
            y = monthly_df['low_savings'].values
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X, y)
            advice_trees['low_savings_rate'] = {
                'model': clf, 'features': ['monthly_savings'], 'condition': clf.predict([[monthly_df['savings'].iloc[-1]]])[0] == 1,
                'tip': "Your savings rate is low. Consider automating a portion of your income to be transferred to savings right after payday.",
            }
        except Exception: pass

    # Heuristic 3: High entertainment spending
    monthly_df['high_entertainment'] = (monthly_df['entertainment'] / monthly_df['income'] > 0.08).astype(int)
    if 'high_entertainment' in monthly_df.columns and monthly_df['high_entertainment'].sum() > 0:
        try:
            X = monthly_df[['entertainment']].values
            y = monthly_df['high_entertainment'].values
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X, y)
            advice_trees['high_entertainment_spend'] = {
                'model': clf, 'features': ['entertainment_spending'], 'condition': clf.predict([[monthly_df['entertainment'].iloc[-1]]])[0] == 1,
                'tip': "Entertainment expenses are significant. Look for free or low-cost activities and events in your area to save money.",
            }
        except Exception: pass

    # Heuristic 4: Volatile balance
    daily_df['balance_change'] = daily_df['running_balance'].pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0)
    volatility_labels = (daily_df['balance_change'].rolling(window=7).mean() > 0.05).astype(int).dropna()
    if not volatility_labels.empty and volatility_labels.sum() > 0:
        try:
            X = daily_df[['balance_change']].dropna().values
            y = volatility_labels.values
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(X, y)
            advice_trees['volatile_balance'] = {
                'model': clf, 'features': ['daily_balance_change'], 'condition': clf.predict([[daily_df['balance_change'].iloc[-1]]])[0] == 1,
                'tip': "Your balance has been volatile recently. This could be due to large, irregular transactions. Try to plan for these to maintain stability.",
            }
        except Exception: pass
        
    return advice_trees

# --- MAIN APP LOGIC
def main():
    """Main function to run the Streamlit app."""
    
    st.title("FinRPG: Your Personal Financial Assistant 🚀")
    
    # --- FILE UPLOAD & INSTRUCTIONS
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is None:
        st.markdown(f"""
            <div style="
                background-color: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(55, 65, 81, 0.7);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin-top: 2rem;
            ">
                <h3 style="color: {SECONDARY_COLOR};">Welcome to FinRPG!</h3>
                <p>Upload a CSV file to begin your financial journey. The app will analyze your spending habits, provide insights, and help you level up your financial game.</p>
                <p>Your CSV file must contain the following columns:</p>
                <code style="background-color: #0F172A; padding: 0.5rem 1rem; border-radius: 6px;">date,category,amount,running_balance</code>
            </div>
            """, unsafe_allow_html=True)
            
        st.download_button(
            label="Download Sample CSV",
            data=SAMPLE_CSV_CONTENT,
            file_name="sample_transactions.csv",
            mime="text/csv",
            key="download_sample_button"
        )
        return

    df = load_and_validate_data(uploaded_file)
    if df is None:
        return
        
    # --- TIME FRAME SELECTION
    st.sidebar.header("Filter Data")
    time_frame_type = st.sidebar.radio("Select timeframe type:", ["All Data", "Year", "Month", "Week"])
    
    df_period = df.copy()

    if time_frame_type == "Year":
        years = df['date'].dt.year.unique().tolist()
        selected_year = st.sidebar.selectbox("Select Year:", years)
        df_period = df[df['date'].dt.year == selected_year]
    elif time_frame_type == "Month":
        months = df['date'].dt.to_period('M').unique().tolist()
        month_str = [m.strftime('%Y-%m') for m in months]
        selected_month_str = st.sidebar.selectbox("Select Month:", month_str)
        selected_month = pd.Period(selected_month_str, freq='M')
        df_period = df[df['date'].dt.to_period('M') == selected_month]
    elif time_frame_type == "Week":
        weeks = df['date'].dt.to_period('W').unique().tolist()
        week_str = [w.strftime('%Y-W%W') for w in weeks]
        selected_week_str = st.sidebar.selectbox("Select Week:", week_str)
        selected_week = pd.Period(selected_week_str, freq='W')
        df_period = df[df['date'].dt.to_period('W') == selected_week]
    
    if df_period.empty:
        st.warning("No data found for the selected timeframe. Please adjust your filters.")
        return

    # --- KPI RAIL
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_balance = df['running_balance'].iloc[-1]
    total_income = df['amount'][df['amount'] > 0].sum()
    total_expenses = abs(df['amount'][df['amount'] < 0].sum())
    
    col1.metric("Current Balance", f"€{current_balance:,.2f}")
    col2.metric("Total Income", f"€{total_income:,.2f}")
    col3.metric("Total Expenses", f"€{total_expenses:,.2f}")
    col4.metric("Net Change", f"€{total_income - total_expenses:,.2f}")
    
    savings = abs(df[df['category'] == 'savings']['amount'].sum())
    savings_rate = (savings / total_income) if total_income > 0 else 0
    col5.metric("Savings Rate", f"{savings_rate:.1%}")

    st.markdown("---")

    # --- TABS FOR VISUALIZATIONS
    tab_charts, tab_projections, tab_gamification = st.tabs(["📊 Financial Overview", "🔮 Balance Projection", "🎮 Fin-RPG"])

    with tab_charts:
        st.header("Financial Overview")
        
        # Balance Chart (Stock Price Style)
        st.subheader("Account Balance & Daily Flow")
        daily_data = compute_daily_data(df)
        ma_data = get_moving_averages(daily_data['end_of_day_balance'], [20, 50])
        
        fig_balance = go.Figure()
        
        fig_balance.add_trace(go.Bar(
            x=daily_data['date'],
            y=daily_data['total_daily_flow'].abs(),
            name='Daily Volume',
            marker_color='rgba(20, 184, 166, 0.4)',
            yaxis='y2'
        ))
        
        fig_balance.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['end_of_day_balance'],
            mode='lines',
            name='Running Balance',
            line=dict(color=PRIMARY_COLOR, width=2)
        ))
        
        fig_balance.add_trace(go.Scatter(
            x=daily_data['date'], y=ma_data['20-day MA'], mode='lines',
            name='20-day MA', line=dict(color='orange', dash='dot')
        ))
        fig_balance.add_trace(go.Scatter(
            x=daily_data['date'], y=ma_data['50-day MA'], mode='lines',
            name='50-day MA', line=dict(color='#A8A29E', dash='dash')
        ))
        
        fig_balance.update_layout(
            title="Account Balance & Transaction Volume",
            xaxis_title="Date",
            yaxis=dict(title="Balance (€)"),
            yaxis2=dict(title="Volume (€)", overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_balance, use_container_width=True)

        st.markdown("---")
        
        # Category Spend Bar Chart
        st.subheader("Spending Breakdown")
        expenses_period_df = df_period[df_period['amount'] < 0]
        category_spend = expenses_period_df.groupby('category')['amount'].sum().abs().reset_index()
        category_spend.rename(columns={'amount': 'total_spent'}, inplace=True)
        category_spend['percent'] = (category_spend['total_spent'] / category_spend['total_spent'].sum()) * 100
        
        fig_spend = px.bar(
            category_spend,
            x='category',
            y='total_spent',
            color='category',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            text=category_spend['percent'].apply(lambda x: f'{x:.1f}%')
        )
        fig_spend.update_layout(
            title="Spending per Category",
            xaxis_title="Category",
            yaxis_title="Total Spent (€)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_spend, use_container_width=True)

    with tab_projections:
        st.header("Balance Projection")
        st.info("🔮 This model forecasts your account balance for the next 3 months based on historical data. Toggle methods in the sidebar.")
        
        st.sidebar.header("Projection Methods")
        proj_methods = {
            'Linear Regression': st.sidebar.checkbox('Linear Regression', value=True),
            'Holt-Winters': st.sidebar.checkbox('Holt-Winters', value=True),
            'Monte Carlo': st.sidebar.checkbox('Monte Carlo', value=True)
        }
        
        if not any(proj_methods.values()):
            st.warning("Please select at least one projection method in the sidebar.")
        else:
            projections = forecast_balance(df, proj_methods)
            
            if projections:
                fig_proj = go.Figure()
                
                fig_proj.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['running_balance'],
                    mode='lines',
                    name='Historical',
                    line=dict(color=SECONDARY_COLOR, width=2)
                ))
                
                future_dates = pd.date_range(start=df['date'].max(), periods=91, freq='D')[1:]
                
                for method, data in projections.items():
                    if 'mean' in data:
                        line_style = dict(dash='dot')
                        if method == 'Ensemble':
                            line_style = dict(color=PRIMARY_COLOR, width=3, dash='solid')
                        
                        fig_proj.add_trace(go.Scatter(
                            x=future_dates, y=data['mean'], mode='lines',
                            name=f'Projected ({method})',
                            line=line_style
                        ))
                        # Confidence interval band
                        fig_proj.add_trace(go.Scatter(
                            x=future_dates,
                            y=data['mean'] + 1.96 * data['std'],
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        fig_proj.add_trace(go.Scatter(
                            x=future_dates,
                            y=data['mean'] - 1.96 * data['std'],
                            fill='tonexty', fillcolor=f'rgba(139, 92, 246, 0.1)' if method == 'Ensemble' else f'rgba(150, 150, 150, 0.1)',
                            mode='lines', line=dict(width=0), showlegend=False
                        ))

                fig_proj.update_layout(
                    title="Account Balance Projection (Next 90 Days)",
                    xaxis_title="Date",
                    yaxis_title="Balance (€)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_proj, use_container_width=True)
                
                # Projection summary
                st.subheader("Projection Summary")
                summary_table = {}
                for method, data in projections.items():
                    if 'mean' in data:
                        summary_table[method] = {
                            "Expected Balance (1 Month)": data['mean'][29],
                            "Expected Balance (3 Months)": data['mean'][89],
                            "Risk (3-month Std)": data['std'][89]
                        }
                st.dataframe(pd.DataFrame(summary_table).T.style.format("€{:.2f}"))
            else:
                st.warning("Forecasting failed. Not enough data or models could not be fitted.")
    
    with tab_gamification:
        st.header("Fin-RPG Gamification")
        
        # Initialize session state for gamification
        if 'total_xp' not in st.session_state:
            st.session_state.total_xp = 0
        if 'achievements' not in st.session_state:
            st.session_state.achievements = []
        if 'current_challenge' not in st.session_state:
            st.session_state.current_challenge = get_daily_challenges()
        
        # XP & Level
        level = st.session_state.total_xp // 100
        xp_progress = st.session_state.total_xp % 100
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.metric("Current Level", f"Level {level + 1}")
        with col_g2:
            st.metric("Total XP", f"{st.session_state.total_xp} XP")

        st.subheader("XP Progress")
        st.progress(xp_progress / 100)
        st.markdown(f"**{xp_progress}/100 XP** to the next level!")

        st.markdown("---")

        # Streaks and Achievements
        daily_balances = df.set_index('date')['running_balance'].resample('D').last().ffill()
        positive_weeks = daily_balances.resample('W').last().diff().dropna() > 0
        streak = 0
        if not positive_weeks.empty:
            streaks = (positive_weeks.groupby((~positive_weeks).cumsum()).cumcount() + 1).max()
            if positive_weeks.iloc[-1]: streak = streaks
        
        st.subheader("Streaks & Achievements")
        st.info(f"🔥 Your streak of weeks with a positive balance is **{streak}**!")
        
        if streak >= 3 and "3-Week Streak" not in st.session_state.achievements:
            st.session_state.total_xp += 50
            st.session_state.achievements.append("3-Week Streak")
            st.balloons()
            st.success("🎉 Achievement Unlocked: 3-Week Streak! (+50 XP)")
        
        if savings_rate > 0.15 and "Savings Champion" not in st.session_state.achievements:
            st.session_state.total_xp += 75
            st.session_state.achievements.append("Savings Champion")
            st.success("💰 Achievement Unlocked: Savings Champion! (+75 XP)")

        st.markdown("---")
        
        # Daily Challenge
        st.subheader("Daily Challenge")
        st.info(f"**Today's challenge:** {st.session_state.current_challenge}")
        if st.button("Complete Challenge"):
            st.session_state.total_xp += 25
            st.session_state.current_challenge = get_daily_challenges()
            st.success("✅ Challenge completed! You earned 25 XP.")
            st.experimental_rerun()
            
        st.markdown("---")

        # Financial Tips
        st.subheader("Actionable Tips")
        st.info("Based on your data, here are some personalized tips to improve your finances.")
        
        advice_trees = create_advice_trees(df_period)
        if advice_trees:
            for tip_id, tip_data in advice_trees.items():
                if tip_data.get('condition', False):
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(30, 41, 59, 0.7);
                        border-left: 4px solid {SECONDARY_COLOR};
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                    ">
                        <p style="font-weight: bold; color: {SECONDARY_COLOR};">Tip: {tip_data['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Show the model's reasoning"):
                        st.code(export_text(tip_data['model'], feature_names=tip_data['features']))
        else:
            st.warning("Not enough data to provide personalized tips.")


if __name__ == "__main__":
    main()
