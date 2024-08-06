import joblib
import pandas as pd
import streamlit as st

# Load models and preprocessor
rf = joblib.load('rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
feature_names_saved = joblib.load('feature_names.pkl')

st.title('Credit Score Prediction with RandomForest')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write(f"Incoming data shape: {df.shape}")

    # Apply preprocessing
    X_transformed = preprocessor.transform(df)
    st.write(f"Transformed data shape: {X_transformed.shape}")

    # Compare feature names
    feature_names_transformed = preprocessor.get_feature_names_out()
    st.write(f"Feature names (saved): {feature_names_saved[:10]} ...")
    st.write(f"Feature names (transformed): {feature_names_transformed[:10]} ...")

    # Predict with RandomForest model
    try:
        predictions_rf = rf.predict(X_transformed)
        st.write("RandomForest Predictions:", predictions_rf.tolist())
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")










