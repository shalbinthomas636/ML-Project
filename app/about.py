
import streamlit as st

from app.styles import page_header


def render(metadata):
    page_header(
        "About the Project",
        "Understanding the technology behind intelligent used car valuation."
    )

    st.markdown("## Used Car Price Prediction")

    st.write("""
    Used Car Price Prediction is a machine-learning application designed
    to estimate the selling price of pre-owned vehicles based on their
    specifications and historical vehicle data.

    The system evaluates multiple regression algorithms and selects
    a model based on its measured prediction performance.
    """)

    st.markdown("### Project Objectives")

    objectives = [
        "Analyze historical used car price data.",
        "Clean and preprocess vehicle specifications.",
        "Train and compare multiple regression algorithms.",
        "Identify a suitable model using evaluation metrics.",
        "Predict vehicle prices from user-entered specifications.",
        "Present results through an interactive web application."
    ]

    for item in objectives:
        st.markdown(f"- {item}")

    st.markdown("### Machine Learning Algorithms")

    algorithms = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Extra Trees Regressor",
        "Gradient Boosting Regressor",
        "AdaBoost Regressor",
        "K-Nearest Neighbors Regressor",
        "Support Vector Regression"
    ]

    left, right = st.columns(2)

    for index, algorithm in enumerate(algorithms):
        with (left if index % 2 == 0 else right):
            st.markdown(f"- {algorithm}")

    st.markdown("### Evaluation Metrics")

    st.write("""
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
    - Mean Absolute Percentage Error (MAPE)
    """)

    st.markdown("### Technology Stack")

    technologies = [
        "Python",
        "Pandas and NumPy",
        "Scikit-learn",
        "Matplotlib and Seaborn",
        "Plotly",
        "Streamlit",
        "Joblib"
    ]

    st.write(" · ".join(technologies))

    st.markdown("### Model Information")

    st.write(
        "Selected algorithm:",
        metadata.get("best_algorithm", "Not available")
    )

    st.info(
        "This project is intended for educational and analytical use. "
        "Predicted prices are estimates and are not guaranteed transaction prices."
    )
