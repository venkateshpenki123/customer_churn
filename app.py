import streamlit as st
import joblib
import pandas as pd
import shap

# Load model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")
st.write("Enter Customer Details")

# Numerical inputs
age = st.number_input("Age", 18, 100, 30)
monthly_income = st.number_input("Monthly Income", 0.0)
monthly_bill = st.number_input("Monthly Bill", 0.0)
internet_usage_gb = st.number_input("Internet Usage (GB)", 0.0)
call_minutes = st.number_input("Call Minutes", 0.0)
support_tickets = st.number_input("Support Tickets", 0)

# Categorical inputs
gender = st.selectbox("Gender", ["Female", "Male"])

city = st.selectbox(
    "City",
    [
        "Faisalabad", "Gujranwala", "Hyderabad", "Islamabad",
        "Karachi", "Lahore", "Multan", "Peshawar",
        "Quetta", "Rawalpindi", "Sialkot"
    ]
)

education = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD", "Primary", "Secondary"]
)

employment = st.selectbox(
    "Employment Status",
    ["Employed", "Retired", "Self-Employed", "Student", "Unemployed"]
)

contract = st.selectbox(
    "Contract Type",
    ["12-Month", "6-Month", "Monthly"]
)

feedback = st.selectbox(
    "Customer Feedback",
    [
        "Billing issues occurred multiple times",
        "Coverage is poor in my location",
        "Customer support was helpful",
        "Frequent call drops in my area",
        "Happy with data packages",
        "Internet speed is very slow",
        "Network quality is good overall",
        "Satisfied with the service",
        "Service is unreliable during peak hours",
        "Too expensive compared to competitors"
    ]
)

if st.button("Predict Churn"):

    data = {
        "age": age,
        "monthly_income": monthly_income,
        "monthly_bill": monthly_bill,
        "internet_usage_gb": internet_usage_gb,
        "call_minutes": call_minutes,
        "support_tickets": support_tickets,
        "signup_year": 2024,
        "signup_month": 1
    }

    # Gender
    for x in ["Female", "Male"]:
        data["gender_" + x] = int(gender == x)

    # City
    cities = [
        "Faisalabad", "Gujranwala", "Hyderabad", "Islamabad",
        "Karachi", "Lahore", "Multan", "Peshawar",
        "Quetta", "Rawalpindi", "Sialkot"
    ]

    for x in cities:
        data["city_" + x] = int(city == x)

    # Education
    for x in ["Bachelor", "Master", "PhD", "Primary", "Secondary"]:
        data["education_level_" + x] = int(education == x)

    # Employment
    for x in ["Employed", "Retired", "Self-Employed", "Student", "Unemployed"]:
        data["employment_status_" + x] = int(employment == x)

    # Contract
    for x in ["12-Month", "6-Month", "Monthly"]:
        data["contract_type_" + x] = int(contract == x)

    # Feedback
    feedback_values = [
        "Billing issues occurred multiple times",
        "Coverage is poor in my location",
        "Customer support was helpful",
        "Frequent call drops in my area",
        "Happy with data packages",
        "Internet speed is very slow",
        "Network quality is good overall",
        "Satisfied with the service",
        "Service is unreliable during peak hours",
        "Too expensive compared to competitors"
    ]

    for x in feedback_values:
        data["customer_feedback_" + x] = int(feedback == x)

    # Create DataFrame
    input_data = pd.DataFrame([data])

    # Get Random Forest from Pipeline
    rf_model = model.named_steps["model"]

    # Match training columns
    input_data = input_data.reindex(
        columns=rf_model.feature_names_in_,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(
        "Churn Probability:",
        round(probability * 100, 2),
        "%"
    )

    if prediction == 1:
        st.error("Prediction: High Churn Risk")
    else:
        st.success("Prediction: Low Churn Risk")

    # SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_data)

    # Get churn SHAP values
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

        if len(values.shape) > 1:
            values = values[:, 1]

    values = values.flatten()

    # Top factors
    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "SHAP": values
    })

    shap_df["Impact"] = shap_df["SHAP"].abs()

    top_factors = shap_df.sort_values(
        "Impact",
        ascending=False
    ).head(3)

    st.write("### Top Factors")

    for _, row in top_factors.iterrows():
        st.write("•", row["Feature"])