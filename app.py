import streamlit as st

from src.predict import load_model, predict_customer


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.caption("Predict whether a telecom customer is likely to churn.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(monthly * max(tenure, 1)),
    )

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

if st.button("🔮 Predict Churn", type="primary"):
    result = predict_customer(customer, model)

    st.divider()

    if result["churn"] == 1:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")

    if result["probability"] is not None:
        probability = result["probability"]
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
        st.progress(probability)
