import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def preprocess_data(data, scaler, model_features, categorical_cols_names, label_encoders_classes):
    """Preprocess data for model prediction"""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Drop columns that were dropped during training
    df = df.drop(columns=[
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeId', 'Attrition'
    ], errors='ignore')

    # Apply Label Encoding for categorical columns FIRST
    for col in categorical_cols_names:
        if col in df.columns:
            le = LabelEncoder()
            le.classes_ = np.array(label_encoders_classes[col])
            try:
                df[col] = le.transform(df[col])
            except ValueError as e:
                # st.warning(f"Warning: Unseen value in {col}. Using default encoding.")
                # Handle unseen labels by using the most frequent class (0)
                df[col] = 0
        else:
            # If categorical column is missing, add it with default value
            df[col] = 0

    # Ensure all features used during training are present
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0

    # Reorder columns to match the training data
    df = df[model_features]

    try:
        df_scaled = pd.DataFrame(
            scaler.transform(df), 
            columns=df.columns, 
            index=df.index
        )
        return df_scaled
    except Exception as e:
        return df

@st.cache_data
def load_employee_data():
    """Load employee data from CSV"""
    try:
        df = pd.read_csv('data/employee_data.csv')
        return df
    except FileNotFoundError:
        st.error("employee_data.csv not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('model/logistic_regression_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        model_features = joblib.load('model/model_features.pkl')
        categorical_cols_names = joblib.load('model/categorical_cols.pkl')
        label_encoders_classes = joblib.load('model/label_encoders_classes.pkl')
        return model, scaler, model_features, categorical_cols_names, label_encoders_classes
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None, None, None

def predict_attrition(data, model, scaler, model_features, categorical_cols_names, label_encoders_classes):
    """Make attrition prediction for given data"""
    processed_data = preprocess_data(data, scaler, model_features, categorical_cols_names, label_encoders_classes)
    
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    
    return prediction, prediction_proba

def create_employee_summary_card(employee_data, prediction, probability):
    """Create a summary card for an employee"""
    attrition_risk = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = "🔴" if prediction == 1 else "🟢"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**Employee ID:** {employee_data.get('EmployeeId', 'N/A')}")
        employee_id = employee_data.get('EmployeeId', 'Unknown')
        employee_name = employee_data.get('Name', f'Employee {employee_id}')
        st.write(f"**Name:** {employee_name}")
        st.write(f"**Department:** {employee_data.get('Department', 'N/A')}")
        st.write(f"**Job Role:** {employee_data.get('JobRole', 'N/A')}")
        st.write(f"**Age:** {employee_data.get('Age', 'N/A')}")
    
    with col2:
        st.write(f"**Years at Company:** {employee_data.get('YearsAtCompany', 'N/A')}")
        st.write(f"**Monthly Income:** ${employee_data.get('MonthlyIncome', 'N/A'):,}")
        st.write(f"**Job Satisfaction:** {employee_data.get('JobSatisfaction', 'N/A')}/4")
        st.write(f"**Work-Life Balance:** {employee_data.get('WorkLifeBalance', 'N/A')}/4")
    
    with col3:
        st.metric(
            label="Attrition Risk",
            value=attrition_risk,
            delta=f"{probability[1]:.1%} probability"
        )
        
        # Progress bar for attrition probability
        st.progress(probability[1])

def main():
    st.title("👥 Employee Attrition Prediction System")
    st.markdown("Select employees from the dataset to predict their likelihood of leaving the company.")
    
    # Load data and model
    df = load_employee_data()
    model_artifacts = load_model_artifacts()
    
    if df is None or any(artifact is None for artifact in model_artifacts):
        st.error("Failed to load required files. Please ensure all model files and employee_data.csv are available.")
        return
    
    model, scaler, model_features, categorical_cols_names, label_encoders_classes = model_artifacts
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filter Employees")
    
    # Department filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    # Job Role filter
    if selected_dept != 'All':
        job_roles = ['All'] + sorted(df[df['Department'] == selected_dept]['JobRole'].unique().tolist())
    else:
        job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
    selected_role = st.sidebar.selectbox("Job Role", job_roles)
    
    # Age range filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Years at company filter
    years_range = st.sidebar.slider(
        "Years at Company",
        min_value=int(df['YearsAtCompany'].min()),
        max_value=int(df['YearsAtCompany'].max()),
        value=(int(df['YearsAtCompany'].min()), int(df['YearsAtCompany'].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
    if selected_role != 'All':
        filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
    
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) & 
        (filtered_df['Age'] <= age_range[1]) &
        (filtered_df['YearsAtCompany'] >= years_range[0]) & 
        (filtered_df['YearsAtCompany'] <= years_range[1])
    ]
    
    st.sidebar.write(f"**Filtered Employees:** {len(filtered_df)}")
    
    st.subheader("📋 Employee Selection")
    
    if len(filtered_df) == 0:
        st.warning("No employees match the current filters.")
        return
    
    # Create a more readable employee list
    employee_display = filtered_df.apply(
        lambda row: f"ID: {row['EmployeeId']} | {row['Department']} - {row['JobRole']} | Age: {row['Age']} | Years: {row['YearsAtCompany']}", 
        axis=1
    ).tolist()
    
    selected_employees = st.multiselect(
        "Select employees to analyze:",
        options=range(len(filtered_df)),
        format_func=lambda x: employee_display[x],
        help="Select one or more employees to predict their attrition risk"
    )
    
    # Prediction Results
    if selected_employees:
        st.subheader("🎯 Attrition Predictions")
        
        predictions_data = []
        
        for idx in selected_employees:
            employee_data = filtered_df.iloc[idx].to_dict()
            
            # Make prediction
            prediction, prediction_proba = predict_attrition(
                employee_data, model, scaler, model_features, 
                categorical_cols_names, label_encoders_classes
            )
            
            predictions_data.append({
                'employee_data': employee_data,
                'prediction': prediction[0],
                'probability': prediction_proba[0]
            })
        
        # Display predictions
        for i, pred_data in enumerate(predictions_data):
            with st.expander(f"Employee {pred_data['employee_data']['EmployeeId']} - {pred_data['employee_data']['Department']}", expanded=True):
                create_employee_summary_card(
                    pred_data['employee_data'],
                    pred_data['prediction'],
                    pred_data['probability']
                )
        
        # Summary visualization
        if len(predictions_data) > 1:
            st.subheader("📈 Prediction Summary")
            
            # Create summary dataframe
            summary_df = pd.DataFrame([
                {
                    'Employee_ID': pred['employee_data']['EmployeeId'],
                    'Department': pred['employee_data']['Department'],
                    'JobRole': pred['employee_data']['JobRole'],
                    'Attrition_Risk': 'High Risk' if pred['prediction'] == 1 else 'Low Risk',
                    'Attrition_Probability': pred['probability'][1],
                    'Age': pred['employee_data']['Age'],
                    'YearsAtCompany': pred['employee_data']['YearsAtCompany'],
                    'MonthlyIncome': pred['employee_data']['MonthlyIncome']
                }
                for pred in predictions_data
            ])
            
            # Risk distribution chart
            fig = px.bar(
                summary_df, 
                x='Employee_ID', 
                y='Attrition_Probability',
                color='Attrition_Risk',
                title="Attrition Risk by Employee",
                labels={'Attrition_Probability': 'Attrition Probability', 'Employee_ID': 'Employee ID'},
                color_discrete_map={'High Risk': '#ff4444', 'Low Risk': '#44ff44'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary table
            st.dataframe(
                summary_df[['Employee_ID', 'Department', 'JobRole', 'Attrition_Risk', 'Attrition_Probability']],
                use_container_width=True
            )
    
    # Batch prediction option
    st.subheader("🔄 Batch Prediction")
    if st.button("Predict All Filtered Employees", type="secondary"):
        if len(filtered_df) > 100:
            st.warning("Large dataset detected. This may take a moment...")
        
        with st.spinner("Making predictions..."):
            batch_predictions = []
            
            for idx, row in filtered_df.iterrows():
                employee_data = row.to_dict()
                prediction, prediction_proba = predict_attrition(
                    employee_data, model, scaler, model_features,
                    categorical_cols_names, label_encoders_classes
                )
                
                batch_predictions.append({
                    'EmployeeId': employee_data['EmployeeId'],
                    'Department': employee_data['Department'],
                    'JobRole': employee_data['JobRole'],
                    'Attrition_Risk': 'High Risk' if prediction[0] == 1 else 'Low Risk',
                    'Attrition_Probability': prediction_proba[0][1],
                    'Age': employee_data['Age'],
                    'YearsAtCompany': employee_data['YearsAtCompany']
                })
        
        batch_df = pd.DataFrame(batch_predictions)
        
        # Show high-risk employees
        high_risk_employees = batch_df[batch_df['Attrition_Risk'] == 'High Risk'].sort_values('Attrition_Probability', ascending=False)
        
        if len(high_risk_employees) > 0:
            st.error(f"⚠️ {len(high_risk_employees)} employees are at high risk of attrition!")
            st.dataframe(high_risk_employees, use_container_width=True)
        else:
            st.success("✅ No high-risk employees found in the current selection!")
        
        # Download results
        csv = batch_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="employee_attrition_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()