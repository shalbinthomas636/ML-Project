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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AUTO VALUE | Used Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

try:
    load_css()

except Exception as error:
    st.warning(f"Custom styling could not be loaded: {error}")


# ============================================================
# APPLICATION SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 15px 5px 22px 5px;">

        <div style="font-size:29px;">
            🚘
        </div>

        <div class="brand-title">
            AUTO VALUE
        </div>

        <div class="brand-subtitle">
            INTELLIGENT CAR VALUATION
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <p style="
        font-size:11px;
        color:#A9C8EA;
        letter-spacing:2px;
        font-weight:600;
    ">
        MAIN MENU
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

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
    options=list(pages.keys()),
    label_visibility="collapsed"
)

page = pages[selected_page]


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        padding:10px 5px;
        color:#C5D9EF;
        font-size:12px;
    ">

        <b>ML POWERED VALUATION</b>

        <br><br>

        Python · Scikit-learn · Streamlit

        <br>

        <span style="color:#8DB8E5;">
            Used Car Price Prediction
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD APPLICATION RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_metadata():
    return load_metadata()


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_comparison():
    return load_comparison()


# ============================================================
# APPLICATION CONTENT
# ============================================================

try:

    with st.spinner("Loading AUTO VALUE application..."):

        model = get_model()

        metadata = get_metadata()

        df = get_dataset()

        comparison = get_comparison()


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "dashboard":

        show_dashboard(
            df,
            comparison,
            metadata
        )


    # ========================================================
    # PRICE PREDICTION
    # ========================================================

    elif page == "prediction":

        show_prediction(
            model,
            metadata,
            df
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    elif page == "comparison":

        show_comparison(
            comparison,
            metadata
        )


    # ========================================================
    # EXPLORE CARS
    # ========================================================

    elif page == "explore":

        show_explore(df)


    # ========================================================
    # DATA ANALYTICS
    # ========================================================

    elif page == "analytics":

        show_analytics(df)


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    elif page == "history":

        show_history()


    # ========================================================
    # ABOUT PROJECT
    # ========================================================

    elif page == "about":

        show_about(metadata)


    # ========================================================
    # APPLICATION FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">

            AUTO VALUE · USED CAR PRICE PREDICTION

            <br>

            Built with Python, Machine Learning and Streamlit

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ERROR HANDLING
# ============================================================

except FileNotFoundError as error:

    st.error("Required application files are missing.")

    st.code(str(error))

    st.info(
        """
        Please verify the following:

        1. Trained model exists inside the models folder.
        2. Model metadata JSON exists.
        3. Cleaned dataset exists.
        4. Required CSV files are available.

        If model files are missing, run:

        python train_model.py
        """
    )


except ImportError as error:

    st.error("Unable to import an application module.")

    st.code(str(error))

    st.info(
        "Check your app folder structure and requirements.txt."
    )


except Exception as error:

    st.error(
        "AUTO VALUE encountered an unexpected error."
    )

    with st.expander("View technical error details"):

        st.exception(error)


# ============================================================
# END OF APPLICATION
# ============================================================
