# kidney_disease_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Detector",
    page_icon="🩺",
    layout="wide"
)

# Load models and artifacts
@st.cache_resource
def load_models():
    try:
        models = {
            'random_forest': pickle.load(open('kidney_disease_model_random_forest.pkl', 'rb')),
            'gradient_boosting': pickle.load(open('kidney_disease_model_gradient_boosting.pkl', 'rb')),
            'xgboost': pickle.load(open('kidney_disease_model_xgboost.pkl', 'rb'))
        }
        scaler = pickle.load(open('kidney_disease_scaler.pkl', 'rb'))
        label_encoders = pickle.load(open('kidney_disease_label_encoders.pkl', 'rb'))
        artifacts = pickle.load(open('kidney_disease_artifacts.pkl', 'rb'))
        
        return models, scaler, label_encoders, artifacts
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def preprocess_input(input_data, label_encoders):
    """Preprocess user input to match training data format"""
    processed_data = input_data.copy()
    
    # Define categorical mappings based on dataset description
    categorical_mappings = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['notpresent', 'present'],
        'ba': ['notpresent', 'present'],
        'htn': ['no', 'yes'],
        'dm': ['no', 'yes'],
        'cad': ['no', 'yes'],
        'appet': ['good', 'poor'],
        'pe': ['no', 'yes'],
        'ane': ['no', 'yes']
    }
    
    for feature, options in categorical_mappings.items():
        if feature in processed_data:
            value = processed_data[feature]
            if value in options:
                if feature in label_encoders:
                    # Handle unseen labels
                    if value in label_encoders[feature].classes_:
                        processed_data[feature] = label_encoders[feature].transform([value])[0]
                    else:
                        # Default to first option if unseen
                        processed_data[feature] = 0
                else:
                    # Fallback mapping
                    processed_data[feature] = options.index(value)
    
    return processed_data

def create_engineered_features(input_dict):
    """Create engineered features similar to training"""
    # Extract basic features in correct order
    basic_features = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 
        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]
    
    features = []
    for feature in basic_features:
        features.append(input_dict.get(feature, 0))
    
    # Extract values for engineered features
    bp = input_dict.get('bp', 70)
    sg = input_dict.get('sg', 1.010)
    bgr = input_dict.get('bgr', 120)
    bu = input_dict.get('bu', 40)
    sc = input_dict.get('sc', 1.0)
    sod = input_dict.get('sod', 140)
    pot = input_dict.get('pot', 4.5)
    al = input_dict.get('al', 0)
    su = input_dict.get('su', 0)
    
    # Add engineered features (same as in training)
    features.append(bp / (sg + 0.001))  # bp_sg_ratio
    features.append(bgr / (bu + 0.001))  # bgr_bu_ratio
    features.append(sod / (pot + 0.001))  # sod_pot_ratio
    features.append(sc / (bu + 0.001))   # sc_bu_ratio
    
    # Risk scores
    features.append(0 if al <= 1 else 1 if al <= 2 else 2 if al <= 3 else 3)  # albumin_risk
    features.append(0 if su <= 1 else 1 if su <= 2 else 2 if su <= 3 else 3)  # sugar_risk
    features.append(0 if sc <= 1.2 else 1 if sc <= 2.0 else 2 if sc <= 3.0 else 3)  # creatinine_risk
    
    return np.array([features])

def main():
    st.title("🩺 Chronic Kidney Disease Detector")
    st.write("This app predicts the risk of chronic kidney disease based on health parameters using machine learning.")
    
    # Load models
    models, scaler, label_encoders, artifacts = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check if all model files are available.")
        st.info("Make sure you have run the training script first to generate the model files.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("Patient Health Parameters")
    
    # Create input sections
    st.sidebar.subheader("Demographic Information")
    age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=45)
    
    st.sidebar.subheader("Blood Pressure & Urine Tests")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80)
        sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.025, value=1.015, step=0.001, format="%.3f")
        al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5], index=0)
        su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5], index=0)
    
    with col2:
        bgr = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=500, value=120)
        bu = st.number_input("Blood Urea (mg/dL)", min_value=10, max_value=200, value=40)
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=15.0, value=1.0, step=0.1)
    
    st.sidebar.subheader("Cell Counts & Electrolytes")
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.5, step=0.1)
        pcv = st.number_input("Packed Cell Volume", min_value=10, max_value=60, value=40)
        wbcc = st.number_input("White Blood Cells (cells/cumm)", min_value=2000, max_value=20000, value=7800)
    
    with col4:
        rbcc = st.number_input("Red Blood Cells (millions/cmm)", min_value=2.0, max_value=8.0, value=4.8, step=0.1)
        sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=160, value=140)
        pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
    
    st.sidebar.subheader("Medical History & Symptoms")
    col5, col6 = st.sidebar.columns(2)
    
    with col5:
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
    
    with col6:
        htn = st.selectbox("Hypertension", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"])
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Normal Ranges:**
    - BP: < 120/80 mm Hg
    - Specific Gravity: 1.005-1.030
    - Blood Glucose: 70-140 mg/dL
    - Blood Urea: 7-20 mg/dL
    - Serum Creatinine: 0.6-1.2 mg/dL
    - Hemoglobin: 13.5-17.5 g/dL (M), 12.0-15.5 g/dL (F)
    - Sodium: 135-145 mEq/L
    - Potassium: 3.5-5.0 mEq/L
    """)
    
    # Prepare input data
    input_data = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 'pcc': pcc,
        'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo,
        'pcv': pcv, 'wbcc': wbcc, 'rbcc': rbcc, 'htn': htn, 'dm': dm, 'cad': cad,
        'appet': appet, 'pe': pe, 'ane': ane
    }
    
    # Main content area
    st.header("Health Parameter Assessment")
    
    # Display current parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Blood Pressure", f"{bp} mm Hg", 
                 delta="Normal" if bp < 120 else "Elevated" if bp < 130 else "High",
                 delta_color="normal" if bp < 120 else "off" if bp < 130 else "inverse")
    
    with col2:
        st.metric("Serum Creatinine", f"{sc:.1f} mg/dL",
                 delta="Normal" if sc <= 1.2 else "High",
                 delta_color="normal" if sc <= 1.2 else "inverse")
    
    with col3:
        st.metric("Blood Urea", f"{bu} mg/dL",
                 delta="Normal" if bu <= 20 else "High",
                 delta_color="normal" if bu <= 20 else "inverse")
    
    with col4:
        st.metric("eGFR Estimate", 
                 f"{(175 * (sc**-1.154) * (age**-0.203) * (0.742 if age > 50 else 1)):.0f} mL/min/1.73m²",
                 delta="Normal" if (175 * (sc**-1.154) * (age**-0.203) * (0.742 if age > 50 else 1)) > 90 else "Check",
                 delta_color="normal" if (175 * (sc**-1.154) * (age**-0.203) * (0.742 if age > 50 else 1)) > 90 else "off")
    
    # Prediction section
    st.header("Kidney Disease Risk Prediction")
    
    if st.button("Check Kidney Disease Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing health parameters..."):
            # Preprocess input
            processed_data = preprocess_input(input_data, label_encoders)
            
            # Create engineered features
            features_engineered = create_engineered_features(processed_data)
            
            # Scale features
            features_scaled = scaler.transform(features_engineered)
            
            # Make predictions with all models
            results = {}
            for model_name, model in models.items():
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
                results[model_name] = {
                    'prediction': prediction[0],
                    'probability': probability[0][1]  # Probability of kidney disease
                }
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        models_display = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting', 
            'xgboost': 'XGBoost'
        }
        
        for i, (model_key, model_display_name) in enumerate(models_display.items()):
            with [col1, col2, col3][i]:
                st.subheader(model_display_name)
                pred = results[model_key]['prediction']
                prob = results[model_key]['probability']
                st.metric(
                    label="Risk Level", 
                    value="High Risk" if pred == 1 else "Low Risk",
                    delta=f"{prob:.1%} probability",
                    delta_color="inverse" if pred == 1 else "normal"
                )
        
        # Show consensus and recommendations
        st.subheader("Overall Assessment")
        positive_count = sum(1 for result in results.values() if result['prediction'] == 1)
        total_models = len(results)
        avg_probability = np.mean([result['probability'] for result in results.values()])
        
        if positive_count == total_models:
            st.error("🔴 **High Kidney Disease Risk Detected**")
            st.warning("""
            **Immediate Recommendations:**
            - Consult with a nephrologist immediately
            - Get comprehensive kidney function tests
            - Monitor blood pressure regularly
            - Follow a kidney-friendly diet (low sodium, controlled protein)
            - Avoid nephrotoxic medications
            - Regular monitoring of kidney function
            """)
        elif positive_count >= total_models / 2:
            st.warning("🟡 **Moderate Kidney Disease Risk**")
            st.info("""
            **Recommendations:**
            - Schedule a consultation with your doctor
            - Get kidney function tests (eGFR, urine albumin)
            - Monitor your health parameters regularly
            - Control blood pressure and diabetes if present
            - Maintain healthy lifestyle habits
            - Stay hydrated
            """)
        else:
            st.success("🟢 **Low Kidney Disease Risk**")
            st.info("""
            **Maintenance Tips:**
            - Continue healthy lifestyle habits
            - Stay hydrated with water
            - Regular exercise and balanced diet
            - Annual health check-ups
            - Monitor blood pressure and blood sugar
            - Avoid excessive pain medication use
            """)
        
        # Risk probability gauge
        st.subheader("Risk Probability")
        st.progress(float(avg_probability))
        st.write(f"Average probability of chronic kidney disease: **{avg_probability:.1%}**")
        
        # Additional medical information
        with st.expander("Understanding Your Results"):
            st.markdown("""
            **About Chronic Kidney Disease (CKD):**
            - CKD means your kidneys are damaged and can't filter blood properly
            - Early detection is crucial for preventing progression
            - Risk factors include diabetes, high blood pressure, family history
            
            **Next Steps:**
            - Discuss these results with your healthcare provider
            - Consider additional tests: eGFR, Urine Albumin-Creatinine Ratio
            - Regular monitoring if you have risk factors
            """)
    
    # Information section
    st.markdown("---")
    st.header("About This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Clinical Parameters Used:**
        - Demographic: Age
        - Blood tests: BP, Blood Glucose, Blood Urea, Serum Creatinine
        - Electrolytes: Sodium, Potassium
        - Hematology: Hemoglobin, PCV, WBCC, RBCC
        - Urine tests: Specific Gravity, Albumin, Sugar
        - Medical history: Hypertension, Diabetes, CAD
        - Symptoms: Appetite, Pedal Edema, Anemia
        """)
    
    with col2:
        st.warning("""
        **Important Medical Disclaimer:**
        This tool is for educational and screening purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        
        **Always consult with qualified healthcare professionals** for medical diagnoses 
        and treatment decisions. The predictions are based on machine learning models 
        and should be verified through proper clinical evaluation and tests.
        """)

if __name__ == "__main__":
    main()
