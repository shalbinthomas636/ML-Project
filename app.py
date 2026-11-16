
import streamlit as st

from app.styles import load_css
from app.common import (
    load_model,
    load_metadata,
    load_dataset,
    load_comparison
)

from app.dashboard import render as show_dashboard
from app.prediction import render as show_prediction
from app.comparison import render as show_comparison
from app.explore_cars import render as show_explore
from app.analytics import render as show_analytics
from app.history import render as show_history
from app.about import render as show_about


st.set_page_config(
    page_title="AUTO VALUE | Used Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

st.sidebar.markdown("""
<div style="padding: 15px 5px 22px 5px;">
    <div style="font-size:29px;">🚘</div>
    <div class="brand-title">AUTO VALUE</div>
    <div class="brand-subtitle">INTELLIGENT CAR VALUATION</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<p style='font-size:11px;color:#A9C8EA;letter-spacing:2px;'>"
    "MAIN MENU</p>",
    unsafe_allow_html=True
)

pages = {
    "🏠  Dashboard": "dashboard",
    "💰  Price Prediction": "prediction",
    "📊  Model Comparison": "comparison",
    "🚗  Explore Cars": "explore",
    "📈  Data Analytics": "analytics",
    "🕘  Prediction History": "history",
    "ℹ️  About Project": "about"
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(pages.keys()),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="padding:10px 5px;color:#C5D9EF;font-size:12px;">
        <b>ML POWERED VALUATION</b><br><br>
        Python · Scikit-learn · Streamlit<br>
        <span style="color:#8DB8E5;">Used Car Price Prediction</span>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    model = load_model()
    metadata = load_metadata()
    df = load_dataset()
    comparison = load_comparison()

    page = pages[selected_page]

    if page == "dashboard":
        show_dashboard(df, comparison, metadata)

    elif page == "prediction":
        show_prediction(model, metadata, df)

    elif page == "comparison":
        show_comparison(comparison, metadata)

    elif page == "explore":
        show_explore(df)

    elif page == "analytics":
        show_analytics(df)

    elif page == "history":
        show_history()

    elif page == "about":
        show_about(metadata)

    st.markdown(
        """
        <div class="footer">
            AUTO VALUE · USED CAR PRICE PREDICTION<br>
            Built with Python, Machine Learning and Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Check your dataset and model files. "
        "If the model files are missing, run python train_model.py first."
    )

except Exception as error:
    st.error("The application encountered an error.")
    st.exception(error)
