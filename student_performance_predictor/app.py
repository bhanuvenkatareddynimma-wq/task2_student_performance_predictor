import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write("Upload a historical dataset to analyze student performance and predict final marks.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.success(f"Dataset Loaded Successfully! Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # ---------------- CLEANING ----------------
    data = data.dropna()

    if len(data) < 2:
        st.error("Dataset should contain at least 2 records.")
        st.stop()

    # ---------------- DATA PREVIEW ----------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(data, use_container_width=True)

    # ---------------- MISSING VALUES ----------------
    st.subheader("🔍 Missing Values")
    st.dataframe(
        data.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        )
    )

    # ---------------- ENCODING ----------------
    for col in data.select_dtypes(include="object").columns:
        if col not in ["Name", "Gender", "Final_Marks"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # ---------------- TARGET CHECK ----------------
    if "Final_Marks" not in data.columns:
        st.error("Dataset must contain 'Final_Marks' column.")
        st.stop()

    # ---------------- FEATURES & TARGET ----------------
    drop_cols = ["Final_Marks", "Name"]

    # Remove Gender completely
    if "Gender" in data.columns:
        drop_cols.append("Gender")

    X = data.drop(columns=drop_cols, errors="ignore")
    y = data["Final_Marks"]

    # ---------------- TRAIN TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---------------- EVALUATION ----------------
    st.subheader("📈 Model Evaluation")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mean_absolute_error(y_test, predictions), 2))
    col2.metric("MSE", round(mean_squared_error(y_test, predictions), 2))
    col3.metric("R² Score", round(r2_score(y_test, predictions), 2))

    # ---------------- ACTUAL VS PREDICTED ----------------
    st.subheader("📉 Actual vs Predicted")

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")

    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("⭐ Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax2)
    st.pyplot(fig2)

    # ---------------- RESULTS ----------------
    st.subheader("📄 Prediction Results")

    results = X_test.copy().reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    results["Actual_Final_Marks"] = y_test_reset
    results["Predicted_Final_Marks"] = predictions.round(2)

    st.dataframe(results, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Predictions",
        csv,
        "student_predictions.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to continue.")