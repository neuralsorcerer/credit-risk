import streamlit as st 
import pandas as pd 
import numpy as np   
import joblib       

# Load the pretrained LightGBM model
model = joblib.load("lgbm.pkl")

# Retrieve the feature names LightGBM expects
try:
    feature_names = model.booster_.feature_name() 
except:
    feature_names = model.feature_name_      

st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide",
) 

st.title("🏦 Credit Risk Prediction")

st.sidebar.header("Applicant Information")

status_map = {
    "< 0 DM (overdrawn)":      "A11",
    "0 – 200 DM":              "A12",
    "≥ 200 DM":                "A13",
    "No checking account":     "A14"
}
history_map = {
    "No credits taken / all paid back":      "A30",
    "All credits at this bank paid back":    "A31",
    "Existing credits paid duly":            "A32",
    "Delay in paying off":                   "A33",
    "Critical account / other credits":      "A34"
}
purpose_map = {
    "Car (new)":        "A40",
    "Car (used)":       "A41",
    "Furniture / equip": "A42",
    "Radio / TV":       "A43",
    "Domestic appliances":"A44",
    "Repairs":          "A45",
    "Education":        "A46",
    "Vacation":         "A47",
    "Retraining":       "A48",
    "Business":         "A49",
    "Others":           "A410"
}
savings_map = {
    "< 100 DM":         "A61",
    "100 – 500 DM":     "A62",
    "500 – 1 000 DM":   "A63",
    "≥ 1 000 DM":       "A64",
    "Unknown / none":   "A65"
}
employment_map = {
    "Unemployed":       "A71",
    "< 1 year":         "A72",
    "1 – 4 years":      "A73",
    "4 – 7 years":      "A74",
    "≥ 7 years":        "A75"
}
personal_map = {
    "Male: divorced/separated":            "A91",
    "Female: divorced/married":           "A92",
    "Male: single":                       "A93",
    "Male: married/widowed":              "A94",
    "Female: single":                     "A95"
}
debtors_map = {
    "None":            "A101",
    "Co-applicant":    "A102",
    "Guarantor":       "A103"
}
property_map = {
    "Real estate":     "A121",
    "Building savings": "A122",
    "Car/other":       "A123",
    "Unknown/none":    "A124"
}
inst_map = {
    "Bank":            "A141",
    "Stores":          "A142",
    "None":            "A143"
}
housing_map = {
    "Rent":            "A151",
    "Own":             "A152",
    "For free":        "A153"
}
telephone_map = {
    "No":              "A191",
    "Yes (registered)": "A192"
}
foreign_map = {
    "Yes":             "A201",
    "No":              "A202"
}
job_map = {
    "Unskilled non-res":  "A171",
    "Unskilled resident": "A172",
    "Skilled employee":   "A173",
    "Management / self":  "A174"
}

# Numeric inputs
duration       = st.sidebar.number_input("Duration (months)", 4, 72, 12)  
credit_amount  = st.sidebar.number_input("Credit Amount", 250, 50000, 1000)
age            = st.sidebar.number_input("Age", 18, 100, 35)

# Categorical inputs
status_code         = status_map[ st.sidebar.selectbox("Checking Account Status", list(status_map.keys())) ]
history_code        = history_map[ st.sidebar.selectbox("Credit History", list(history_map.keys())) ]
purpose_code        = purpose_map[ st.sidebar.selectbox("Purpose", list(purpose_map.keys())) ]
savings_code        = savings_map[ st.sidebar.selectbox("Savings Account/Bonds", list(savings_map.keys())) ]
employment_code     = employment_map[ st.sidebar.selectbox("Employment Since", list(employment_map.keys())) ]
personal_code       = personal_map[ st.sidebar.selectbox("Personal Status & Sex", list(personal_map.keys())) ]
debtors_code        = debtors_map[ st.sidebar.selectbox("Other Debtors/Guarantors", list(debtors_map.keys())) ]
property_code       = property_map[ st.sidebar.selectbox("Property", list(property_map.keys())) ]
inst_code           = inst_map[ st.sidebar.selectbox("Other Installment Plans", list(inst_map.keys())) ]
housing_code        = housing_map[ st.sidebar.selectbox("Housing", list(housing_map.keys())) ]
telephone_code      = telephone_map[ st.sidebar.selectbox("Telephone", list(telephone_map.keys())) ]
foreign_code        = foreign_map[ st.sidebar.selectbox("Foreign Worker", list(foreign_map.keys())) ]
job_code            = job_map[ st.sidebar.selectbox("Job", list(job_map.keys())) ]

# Feature engineering
log_credit_amount = np.log1p(credit_amount)                     # log(1+x) transform 
monthly_payment   = credit_amount / duration                    # simple ratio 

# Assemble raw DataFrame
df = pd.DataFrame({
    "Duration": [duration],
    "CreditAmount": [credit_amount],
    "Age": [age],
    "LogCreditAmount": [log_credit_amount],
    "MonthlyPayment": [monthly_payment],
    "Status":        [status_code],
    "CreditHistory": [history_code],
    "Purpose":       [purpose_code],
    "Savings":       [savings_code],
    "Employment":    [employment_code],
    "PersonalStatus":[personal_code],
    "OtherDebtors":  [debtors_code],
    "Property":      [property_code],
    "OtherInstallmentPlans":[inst_code],
    "Housing":       [housing_code],
    "Telephone":     [telephone_code],
    "ForeignWorker": [foreign_code],
    "Job":           [job_code]
})

# One-hot encode categoricals
df_encoded = pd.get_dummies(df) 

# Align features to model expectation
for feat in feature_names:
    if feat not in df_encoded.columns:
        df_encoded[feat] = 0
df_model = df_encoded[feature_names]

# Predict probabilities
prob_bad = model.predict_proba(df_model)[:, 1][0]  # Probability of class “bad” 
label = "❌ Bad Risk" if prob_bad >= 0.5 else "✅ Good Risk"

# Display KPIs
col1, col2 = st.columns(2)
col1.metric("Bad-Risk Probability", f"{prob_bad:.1%}")
col2.metric("Prediction", label)

# Detailed views in expanders
with st.expander("Input Features"):
    st.dataframe(df, use_container_width=True)

with st.expander("Encoded Model Features"):
    st.dataframe(df_model, use_container_width=True)

# Download result
df["BadRiskProb"] = prob_bad
df["Prediction"]  = label
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Prediction",
    data=csv,
    file_name="prediction.csv",
    mime="text/csv"
)
