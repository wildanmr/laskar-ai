import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes ML Model Interface",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_ENDPOINT = "http://700070007.xyz:8080/invocations"
SCALER_URL = "https://github.com/wildanmr/Eksperimen_SML_Wildan-Mufid-Ramadhan/releases/latest/download/scaler.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Feature descriptions for better UX
FEATURE_DESCRIPTIONS = {
    "HighBP": "High Blood Pressure (0=No, 1=Yes)",
    "HighChol": "High Cholesterol (0=No, 1=Yes)",
    "CholCheck": "Cholesterol Check (0=No, 1=Yes)",
    "BMI": "Body Mass Index",
    "Smoker": "Smoker (0=No, 1=Yes)",
    "Stroke": "Had Stroke (0=No, 1=Yes)",
    "HeartDiseaseorAttack": "Heart Disease or Attack (0=No, 1=Yes)",
    "PhysActivity": "Physical Activity (0=No, 1=Yes)",
    "Fruits": "Consume Fruits (0=No, 1=Yes)",
    "Veggies": "Consume Vegetables (0=No, 1=Yes)",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption (0=No, 1=Yes)",
    "AnyHealthcare": "Any Healthcare (0=No, 1=Yes)",
    "NoDocbcCost": "No Doctor because of Cost (0=No, 1=Yes)",
    "GenHlth": "General Health (1=Excellent, 5=Poor)",
    "MentHlth": "Mental Health Days (0-30)",
    "PhysHlth": "Physical Health Days (0-30)",
    "DiffWalk": "Difficulty Walking (0=No, 1=Yes)",
    "Sex": "Sex (0=Female, 1=Male)",
    "Age": "Age Category (1-13)",
    "Education": "Education Level (1-6)",
    "Income": "Income Level (1-8)"
}

def make_prediction_request(endpoint, data):
    """Make a single prediction request"""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        return {
            'status_code': response.status_code,
            'response': response.json() if response.status_code == 200 else response.text,
            'response_time': response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            'status_code': 0,
            'response': str(e),
            'response_time': None
        }

def stress_test_worker(endpoint, data, num_requests):
    """Worker function for stress testing"""
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_prediction_request, endpoint, data) for _ in range(num_requests)]
        
        for future in as_completed(futures):
            result = future.result()
            result['timestamp'] = time.time() - start_time
            results.append(result)
    
    return results

def create_sample_data():
    """Create sample normalized data for testing"""
    return {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": [
                [
                    0.8663875715513917, 0.9391870880454268, 0.1607726785944992,
                    1.4051871930524469, -0.9645001347013894, -0.2607083092123164,
                    -0.4215251042181655, -1.5148281085566382, 0.8069038625481344,
                    0.524704116849121, -0.2132943011689738, 0.219817012721681,
                    -0.3261342499282417, 1.02560203011439, 2.084762367037103,
                    0.7944237442944474, -0.5906117333900115, -0.916407578904008,
                    -0.5611929829056437, 0.0968733813373412, 0.6199086151085422
                ]
            ]
        }
    }

@st.cache_resource
def download_and_load_scaler():
    """Download and load the scaler from GitHub"""
    try:
        if not os.path.exists(SCALER_FILE):
            st.info("Downloading scaler from GitHub...")
            response = requests.get(SCALER_URL)
            response.raise_for_status()
            
            with open(SCALER_FILE, 'wb') as f:
                f.write(response.content)
            st.success("✅ Scaler downloaded successfully!")
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = joblib.load(f)
        
        return scaler
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        return None

def normalize_features_with_scaler(features, scaler):
    """Normalize features using the actual scaler from training"""
    if scaler is None:
        st.error("Scaler not available. Using raw features (may cause poor predictions).")
        return features
    
    try:
        # Convert to numpy array and reshape for single sample
        features_array = np.array(features).reshape(1, -1)
        
        # Apply the same normalization as used in training
        normalized_features = scaler.transform(features_array)
        
        return normalized_features[0].tolist()
    except Exception as e:
        st.error(f"Error normalizing features: {str(e)}")
        return features

# Sidebar
st.sidebar.title("🩺 Diabetes ML Model")
st.sidebar.markdown("### Configuration")

# Load scaler
with st.sidebar:
    st.markdown("### Scaler Status")
    scaler = download_and_load_scaler()
    if scaler is not None:
        st.success("✅ Scaler loaded")
    else:
        st.error("❌ Scaler failed to load")

# Endpoint configuration
endpoint = st.sidebar.text_input("Model Endpoint", value=DEFAULT_ENDPOINT)

# Mode selection
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Prediction", "Stress Testing", "Batch Prediction"]
)

# Main content
st.title("Diabetes Prediction ML Model Interface")
st.markdown("This application provides an interface to interact with the diabetes prediction ML model.")

if mode == "Single Prediction":
    st.header("🔍 Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Conditions")
        high_bp = st.selectbox("High Blood Pressure", [0, 1], help=FEATURE_DESCRIPTIONS["HighBP"])
        high_chol = st.selectbox("High Cholesterol", [0, 1], help=FEATURE_DESCRIPTIONS["HighChol"])
        chol_check = st.selectbox("Cholesterol Check", [0, 1], help=FEATURE_DESCRIPTIONS["CholCheck"])
        smoker = st.selectbox("Smoker", [0, 1], help=FEATURE_DESCRIPTIONS["Smoker"])
        stroke = st.selectbox("Had Stroke", [0, 1], help=FEATURE_DESCRIPTIONS["Stroke"])
        heart_disease = st.selectbox("Heart Disease/Attack", [0, 1], help=FEATURE_DESCRIPTIONS["HeartDiseaseorAttack"])
        
        st.subheader("Lifestyle")
        phys_activity = st.selectbox("Physical Activity", [0, 1], help=FEATURE_DESCRIPTIONS["PhysActivity"])
        fruits = st.selectbox("Consume Fruits", [0, 1], help=FEATURE_DESCRIPTIONS["Fruits"])
        veggies = st.selectbox("Consume Vegetables", [0, 1], help=FEATURE_DESCRIPTIONS["Veggies"])
        heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1], help=FEATURE_DESCRIPTIONS["HvyAlcoholConsump"])
        
    with col2:
        st.subheader("Physical Characteristics")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help=FEATURE_DESCRIPTIONS["BMI"])
        gen_health = st.slider("General Health", 1, 5, 3, help=FEATURE_DESCRIPTIONS["GenHlth"])
        ment_health = st.slider("Mental Health Days", 0, 30, 0, help=FEATURE_DESCRIPTIONS["MentHlth"])
        phys_health = st.slider("Physical Health Days", 0, 30, 0, help=FEATURE_DESCRIPTIONS["PhysHlth"])
        diff_walk = st.selectbox("Difficulty Walking", [0, 1], help=FEATURE_DESCRIPTIONS["DiffWalk"])
        
        st.subheader("Demographics")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=FEATURE_DESCRIPTIONS["Sex"])
        age = st.slider("Age Category", 1, 13, 7, help=FEATURE_DESCRIPTIONS["Age"])
        education = st.slider("Education Level", 1, 6, 4, help=FEATURE_DESCRIPTIONS["Education"])
        income = st.slider("Income Level", 1, 8, 5, help=FEATURE_DESCRIPTIONS["Income"])
        
        st.subheader("Healthcare Access")
        any_healthcare = st.selectbox("Any Healthcare", [0, 1], help=FEATURE_DESCRIPTIONS["AnyHealthcare"])
        no_doc_cost = st.selectbox("No Doctor bc Cost", [0, 1], help=FEATURE_DESCRIPTIONS["NoDocbcCost"])
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        # Collect all features
        features = [
            high_bp, high_chol, chol_check, bmi, smoker, stroke,
            heart_disease, phys_activity, fruits, veggies, heavy_alcohol,
            any_healthcare, no_doc_cost, gen_health, ment_health,
            phys_health, diff_walk, sex, age, education, income
        ]
        
        # Normalize features using the actual scaler
        normalized_features = normalize_features_with_scaler(features, scaler)
        
        # Prepare request data
        request_data = {
            "dataframe_split": {
                "columns": FEATURE_COLUMNS,
                "data": [normalized_features]
            }
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            result = make_prediction_request(endpoint, request_data)
        
        # Display results
        if result['status_code'] == 200:
            st.success("✅ Prediction completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Response Time", f"{result['response_time']:.3f}s")
            
            with col2:
                prediction = result['response']['predictions'][0] if 'predictions' in result['response'] else result['response']
                st.metric("Prediction", f"{prediction}")
            
            with col3:
                risk_level = "High Risk" if float(prediction) > 0.5 else "Low Risk"
                st.metric("Risk Level", risk_level)
                
            # Show raw response
            with st.expander("Raw Response"):
                st.json(result['response'])
                
        else:
            st.error(f"❌ Prediction failed: {result['response']}")

elif mode == "Stress Testing":
    st.header("⚡ Stress Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_requests = st.slider("Number of Concurrent Requests", 1, 100, 10)
        use_sample_data = st.checkbox("Use Sample Data", value=True)
        
    with col2:
        if not use_sample_data:
            st.warning("Custom data input for stress testing")
            # You can add custom data input here
    
    if st.button("🚀 Start Stress Test", type="primary"):
        # Prepare test data
        if use_sample_data:
            test_data = create_sample_data()
        else:
            # Use custom data if implemented
            test_data = create_sample_data()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start stress test
        start_time = time.time()
        status_text.text(f"Starting stress test with {num_requests} requests...")
        
        # Run stress test
        results = stress_test_worker(endpoint, test_data, num_requests)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        progress_bar.progress(1.0)
        status_text.text("Stress test completed!")
        
        # Analyze results
        successful_requests = [r for r in results if r['status_code'] == 200]
        failed_requests = [r for r in results if r['status_code'] != 200]
        
        response_times = [r['response_time'] for r in successful_requests if r['response_time']]
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", num_requests)
        
        with col2:
            st.metric("Successful", len(successful_requests))
        
        with col3:
            st.metric("Failed", len(failed_requests))
        
        with col4:
            success_rate = (len(successful_requests) / num_requests) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        if response_times:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Response Time", f"{np.mean(response_times):.3f}s")
            
            with col2:
                st.metric("Min Response Time", f"{np.min(response_times):.3f}s")
            
            with col3:
                st.metric("Max Response Time", f"{np.max(response_times):.3f}s")
            
            with col4:
                throughput = len(successful_requests) / total_time
                st.metric("Throughput", f"{throughput:.1f} req/s")
        
        # Visualizations
        if response_times:
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time distribution
                fig_hist = px.histogram(
                    x=response_times,
                    title="Response Time Distribution",
                    labels={'x': 'Response Time (seconds)', 'y': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Response time over time
                timestamps = [r['timestamp'] for r in successful_requests if r['response_time']]
                fig_line = px.line(
                    x=timestamps,
                    y=response_times,
                    title="Response Time Over Time",
                    labels={'x': 'Time (seconds)', 'y': 'Response Time (seconds)'}
                )
                st.plotly_chart(fig_line, use_container_width=True)
        
        # Detailed results
        with st.expander("Detailed Results"):
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

elif mode == "Batch Prediction":
    st.header("📊 Batch Prediction")
    
    st.markdown("Upload a CSV file with the required features for batch prediction.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Validate columns
            missing_columns = set(FEATURE_COLUMNS) - set(df.columns)
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
            else:
                st.success("✅ All required columns present")
                
                if st.button("🔮 Make Batch Predictions", type="primary"):
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        # Get raw features
                        raw_features = row[FEATURE_COLUMNS].tolist()
                        
                        # Normalize features using the actual scaler
                        normalized_features = normalize_features_with_scaler(raw_features, scaler)
                        
                        # Prepare request data
                        request_data = {
                            "dataframe_split": {
                                "columns": FEATURE_COLUMNS,
                                "data": [normalized_features]
                            }
                        }
                        
                        # Make prediction
                        result = make_prediction_request(endpoint, request_data)
                        
                        if result['status_code'] == 200:
                            pred = result['response']['predictions'][0] if 'predictions' in result['response'] else result['response']
                            predictions.append(pred)
                        else:
                            predictions.append(None)
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Add predictions to dataframe
                    df['Prediction'] = predictions
                    df['Risk_Level'] = df['Prediction'].apply(lambda x: 'High Risk' if x and float(x) > 0.5 else 'Low Risk' if x else 'Error')
                    
                    st.subheader("Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    valid_predictions = [p for p in predictions if p is not None]
                    if valid_predictions:
                        high_risk_count = sum(1 for p in valid_predictions if float(p) > 0.5)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(valid_predictions))
                        with col2:
                            st.metric("High Risk", high_risk_count)
                        with col3:
                            st.metric("Low Risk", len(valid_predictions) - high_risk_count)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Note:** This application interfaces with the diabetes ML model using the official scaler from the GitHub repository.")
st.markdown("**Docker Command:** `docker pull ghcr.io/wildanmr/diabetes-ml-mlflow:latest`")
st.markdown("**Scaler Source:** [GitHub Repository](https://github.com/wildanmr/Eksperimen_SML_Wildan-Mufid-Ramadhan)")

# Display scaler info if available
if scaler is not None:
    with st.expander("Scaler Information"):
        st.write("**Scaler Type:**", type(scaler).__name__)
        if hasattr(scaler, 'mean_'):
            st.write("**Features Mean:**", scaler.mean_)
        if hasattr(scaler, 'scale_'):
            st.write("**Features Scale:**", scaler.scale_)
        if hasattr(scaler, 'feature_names_in_'):
            st.write("**Feature Names:**", scaler.feature_names_in_)