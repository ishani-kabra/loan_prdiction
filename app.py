import joblib 
import streamlit as st
import pandas as pd 

model=joblib.load('loan_model_pipeline.pkl')
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="$",
    layout="wide"
)

st.title("Loan Approval Prediction")

st.divider()

st.subheader("Applicant's Infomation")
col1,col2=st.columns(2)

with col1:

    no_of_dependents=st.number_input("Number of Dependents",min_value=0,max_value=20,value=2)
    education=st.selectbox('Education_level',options=['Graduate','Not Graduate'])
    self_employed=st.selectbox('Self_Employed',options=['Yes','No'])
    income_annum = st.number_input(
        "Annual Income",
        min_value=0,
        value=5000000,
        step=100000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150000,
        step=10000
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        max_value=50,
        value=15
    )
with col2:
    cibil_score=st.number_input("Cibil Score ",min_value=0,max_value=900,step=50)
    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=5000000,
        step=100000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=3000000,
        step=100000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=5000000,
        step=100000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=3000000,
        step=100000
    )
if st.button("Predict Loan Status", type="primary"):

    # Calculate total assets
    total_asset = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )

    # Create input dataframe
    input_data = pd.DataFrame({
        " no_of_dependents": [no_of_dependents],
        " education": [education],
        " self_employed": [self_employed],
        " income_annum": [income_annum],
        " loan_amount": [loan_amount],
        " loan_term": [loan_term],
        " cibil_score": [cibil_score],
        " residential_assets_value": [residential_assets_value],
        " commercial_assets_value": [commercial_assets_value],
        " luxury_assets_value": [luxury_assets_value],
        " bank_asset_value": [bank_asset_value],
        "total_asset": [total_asset]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    classes = model.classes_

    approved_probability = probability[
     list(classes).index(" Approved")
]

    if prediction == " Approved":
     st.success("### ✅ Loan Approved")
    else:
     st.error("### ❌ Loan Rejected")

    st.metric(
     "Approval Probability",
     f"{approved_probability * 100:.2f}%"
)
    # # Display result
    # if prediction == "Approved":

    #     st.success("### Loan Approved")

    #     st.metric(
    #         "Approval Probability",
    #         f"{approved_probability * 100:.2f}%"
    #     )

    # else:

    #     st.error("###  Loan Rejected")

    #     st.metric(
    #         "Approval Probability",
    #         f"{approved_probability * 100:.2f}%"
    #     )