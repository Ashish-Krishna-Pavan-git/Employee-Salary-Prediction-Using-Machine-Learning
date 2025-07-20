import streamlit as st
import pandas as pd
import joblib
import os
import shap
import json
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] { background-color: rgba(0, 0, 0, 0.2); }
</style>
""", unsafe_allow_html=True)


# --- Backend: Load Assets ---
MODEL_DIR = "Models"
ASSETS_DIR = "Assets"

@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_').lower()}_pipeline.pkl")
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data(file_name):
    path = os.path.join(ASSETS_DIR, file_name)
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(path)
        elif file_name.endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
    except FileNotFoundError:
        return None

# --- UI Sidebar ---
with st.sidebar:
    st.title("Salary Prediction")
    page = option_menu(
        None, ["Prediction App", "Model Performance"],
        icons=["cash-coin", "bar-chart-line"], menu_icon="cast", default_index=0
    )
    st.info("Navigate between the prediction tool and the model performance analysis.")


# --- Prediction App Page ---
if page == "Prediction App":
    st.header("💵 Predict an Individual's Salary")
    
    MODEL_FILES = {
        "XGBoost": "xgboost_pipeline.pkl", "Random Forest": "random_forest_pipeline.pkl",
        "MLP Classifier": "mlp_classifier_pipeline.pkl", "Logistic Regression": "logistic_regression_pipeline.pkl",
        "K-Nearest Neighbors": "k-nearest_neighbors_pipeline.pkl"
    }
    selected_model_name = st.selectbox("Choose a Model for Prediction", list(MODEL_FILES.keys()))

    # Input fields
    workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov']
    marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed']
    occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv']
    native_country_options = ['United-States', 'Mexico', 'Others', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Dominican-Republic']
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 65, 35)
        workclass = st.selectbox("Work Class", workclass_options)
        marital_status = st.selectbox("Marital Status", marital_status_options)
    with col2:
        educational_num = st.slider("Years of Education", 1, 16, 10)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        gender = st.selectbox("Gender", ['Male', 'Female'])
    
    occupation = st.selectbox("Occupation", occupation_options)
    native_country = st.selectbox("Native Country", native_country_options)

    if st.button("Predict Income", type="primary"):
        model_pipeline = load_model(selected_model_name)
        
        if model_pipeline:
            input_data = pd.DataFrame({
                'age': [age],
                'educational-num': [educational_num],
                'hours-per-week': [hours_per_week],
                'net_capital': [0],
                'workclass': [workclass],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'gender': [gender],
                'native-country': [native_country]
            })
            
            # The debugging line has been removed from here.

            try:
                prediction = model_pipeline.predict(input_data)[0]
                prediction_proba = model_pipeline.predict_proba(input_data)[0]
                
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.success("Predicted Income: **> $50K**")
                    else:
                        st.info("Predicted Income: **<= $50K**")
                with col2:
                    prob = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                    st.metric(label="Confidence", value=f"{prob:.2%}")
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Model Performance Page ---
if page == "Model Performance":
    st.header("📊 Model Performance & Selection")
    
    performance_df = load_data("model_performance.csv")
    roc_data = load_data("roc_data.json")
    
    if performance_df is not None:
        best_model = performance_df.iloc[0]
        st.subheader("Best Performing Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model['Model'])
        with col2:
            st.metric("F1 Score", f"{best_model['F1 Score']:.4f}")
        with col3:
            st.metric("ROC AUC", f"{best_model['ROC AUC']:.4f}")

        st.markdown("---")
        st.subheader("All Model Scores")
        st.dataframe(performance_df.style.format({
            'Accuracy': '{:.4f}',
            'F1 Score': '{:.4f}',
            'ROC AUC': '{:.4f}'
        }))
        
        st.subheader("Why was the Best Model Chosen?")
        st.markdown(f"""
        The **{best_model['Model']}** model was selected as the best performer based on its **F1 Score**. This metric is crucial because the dataset is imbalanced. The F1 score provides a balance between Precision and Recall, making it the most reliable indicator of a model's true performance on this task.
        """)
        
        if roc_data:
            st.subheader("ROC Curve Comparison")
            fig = go.Figure()
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            
            for name, data in roc_data.items():
                if name in performance_df['Model'].values:
                    auc = performance_df.loc[performance_df['Model'] == name, 'ROC AUC'].iloc[0]
                    fig.add_trace(go.Scatter(x=data['fpr'], y=data['tpr'], name=f"{name} (AUC={auc:.4f})", mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1), xaxis=dict(constrain='domain'),
                legend_title_text='Models'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.error("Performance data not found. Please run the training notebook to generate the `Assets` folder and its files.")
