import streamlit as st
import joblib
import pandas as pd

# Load the model
@st.cache_resource
def load_nrf_cat_model():
    return joblib.load("category_nrf_isotonic_13 features_6 ds_model.joblib")

@st.cache_resource
def load_nrf_cont_model():
    return joblib.load("continous_nrf_sigmoid_11 features_3 ds_model.joblib")

nrf_cat_model = load_nrf_cat_model()
nrf_cont_model = load_nrf_cont_model()

# Sidebar for Model Selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest Model", "Logistic Regression Model"])

# Page title that reflects the selected model
st.title(f"Patient Details - {model_choice}")

if model_choice == "Random Forest Model":
    # Create three columns for the sections
    col1, spacer1, col2, spacer2, col3 = st.columns([2, 0.5, 2, 0.5, 2])

    # Section 1: Medical Conditions (Categorical/Binary Features)
    with col1:
        st.header("Conditions")
        atrial_fibrillation = st.checkbox("Atrial Fibrillation")
        mi_nstemi = st.checkbox("MI or NSTEMI")
        pvd = st.checkbox("Peripheral Vascular Disease (PVD)")
        cva = st.checkbox("Cerebrovascular Accident (CVA)")
        dementia = st.checkbox("Dementia")
        adl_dependent = st.checkbox("ADL Dependent")
        heart_failure = st.checkbox('Heart Failure')

    # Section 2: Age & Laboratory Data
    with col2:
        st.header("Lab Data")

        # eGFR level
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, step=0.1, format="%.2f")
        egfr_15_abv = int(egfr >= 15)
        
        # Albumin level
        albumin = st.number_input("Albumin (g/L)", min_value=0.0, step=0.1, format="%.2f")
        albumin_35_abv = int(albumin >= 35)
        
        # Hemoglobin level
        haemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.2f")
        haemoglobin_10_abv = int(haemoglobin >= 10)
        
        # Phosphate Inorganic level
        phosphate = st.number_input("Phosphate Inorganic, serum (mmol/L)", min_value=0.0, step=0.1, format="%.2f")
        phosphate_1_6_abv = int(phosphate >= 1.6)

    # Section 3: Renal Function & Comorbidity Index
    with col3:
        st.header("Others")

        # Age categories
        age = st.number_input("Age", min_value=0, step=1)
        age_76_80 = int(76 <= age <= 80)
        age_85_abv = int(age >= 85)

        # CCI score
        cci = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, step=1)
        cci_abv_5 = int(cci > 5)



_, center_col, _ = st.columns([1, 1, 1])  # Adjust widths if needed
# Create a placeholder for the output
output_placeholder = st.empty()

with center_col:
# Display model-specific options or calculations after submission
# Submit button
    if st.button("Submit Data"):
        # Prepare the input data for the model as a dictionary
        input_data_categorical = {
            'Atrial fibrillation': int(atrial_fibrillation),
            'MI or NSTEMI': int(mi_nstemi),
            'PVD': int(pvd),
            'CVA': int(cva),    
            'Dementia': int(dementia),
            'ADL Dependent': int(adl_dependent),
            'age_76-80': age_76_80,
            'age_85 abv': age_85_abv,
            'albumin_35 abv': albumin_35_abv,
            'Haemoglobin >= 10': haemoglobin_10_abv,
            'Phosphate Inorganic, serum >= 1.6': phosphate_1_6_abv,
            'eGFR (CKD-EPI)_15 abv': egfr_15_abv,
            'cci_abv 5': cci_abv_5
        }

        input_categorical_df = pd.DataFrame([input_data_categorical])

        input_data_continuous = {
            'Age' : age,
            'Atrial fibrillation': int(atrial_fibrillation),
            'MI or NSTEMI': int(mi_nstemi),
            'CVA': int(cva),    
            'Chronic Heart Failure (merged)' : int(heart_failure),
            'Albumin, serum' : albumin,
            'Haemoglobin' : haemoglobin,
            'Phosphate Inorganic, serum' : phosphate,
            'eGFR (CKD-EPI)': egfr,
            'ADL Dependent' : int(adl_dependent),
            'cci' : cci
        }

        input_continuous_df = pd.DataFrame([input_data_continuous])

        if model_choice == "Random Forest Model":
            # Insert model-specific logic for Random Forest here
            # Make prediction
            prob_categorized = nrf_cat_model.predict_proba(input_categorical_df)[0, 1]
            prob_continuous = nrf_cont_model.predict_proba(input_continuous_df)[0, 1]

            # Create output table with centered alignment
            results_df = pd.DataFrame(
                {
                    "Model Based on Categorized Values": [f"{prob_categorized:.2%}"],
                    "Model Based on Continuous Values": [f"{prob_continuous:.2%}"]
                }
            )

             # Display results in the placeholder
            output_placeholder.success(
                f"""
                #### Predicted Probability:
                
                - **Model based on categorized values**: {prob_categorized:.2%}
                - **Model based on continuous values**: {prob_continuous:.2%}
                """
            )

            # Example: prediction = random_forest_model.predict(input_data)

        elif model_choice == "Logistic Regression Model":
            # Insert model-specific logic for Logistic Regression here
            st.success(f"Data submitted to Logistic Regression Model: NOT CODED")
            # Example: prediction = logistic_regression_model.predict(input_data)
