
import streamlit as st


def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F3F7FC;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102B50 0%, #123D70 100%);
        border-right: 0;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.055);
        border-radius: 9px;
        padding: 9px 12px;
        margin-bottom: 5px;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.15);
    }

    .brand-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: 1px;
        color: white;
        margin-bottom: 0;
    }

    .brand-subtitle {
        font-size: 10px;
        color: #A9C8EA;
        letter-spacing: 2px;
        margin-top: 0;
    }

    .hero {
        background: linear-gradient(120deg, #102B50, #1766B5, #278BEB);
        padding: 34px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(21, 91, 158, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #D7E9FC;
        font-size: 14px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E3EBF5;
        border-radius: 15px;
        padding: 19px;
        min-height: 120px;
        box-shadow: 0 5px 20px rgba(28, 70, 115, 0.045);
    }

    .metric-label {
        color: #71839B;
        font-size: 12px;
        font-weight: 600;
    }

    .metric-value {
        color: #153D6B;
        font-size: 27px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-note {
        color: #8292A8;
        font-size: 11px;
        margin-top: 3px;
    }

    .section-heading {
        color: #163B65;
        font-size: 21px;
        font-weight: 800;
        margin: 22px 0 12px 0;
    }

    .info-card {
        background: white;
        border: 1px solid #E2EAF4;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .stButton > button {
        background: linear-gradient(110deg, #1766B5, #2589E8);
        color: white;
        border: 0;
        border-radius: 9px;
        padding: 10px 20px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #104C8C;
        color: white;
        border: 0;
    }

    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #B9D3ED;
        color: #17558F;
        background: #EDF6FF;
        font-weight: 700;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E3EBF5;
        border-radius: 12px;
        overflow: hidden;
    }

    .price-result {
        background: linear-gradient(120deg, #102B50, #1766B5);
        color: white;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
    }

    .price-result .amount {
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .footer {
        text-align: center;
        color: #8393A8;
        font-size: 11px;
        padding: 25px 0 10px 0;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 23px;
        }
        .hero h1 {
            font-size: 25px;
        }
        .metric-value {
            font-size: 22px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
